"""Outer-held-out task/stroke/time explanations, geometry and temporal controls."""
import argparse,copy,time
import torch
from scipy.special import expit
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from raw_temporal_common import *
from train_raw_temporal import load_checkpoint,tensor_person,DEV,CONFIG,seed_all
from diagramo.temporal.dataset import load_person
from diagramo.temporal.explain import converged_ig,task_mass

def segment_mapping(person,tasks):
    rows=[];offset=0
    for ti,inv in enumerate(person['inventories']):
        down={};counter=-1
        for run in np.unique(inv[:,3]).astype(int):
            first=inv[inv[:,3]==run][0]
            if first[4]==1:counter+=1;down[run]=counter
        for si in np.unique(inv[:,2]).astype(int):
            d=inv[inv[:,2]==si];a,b=d[0],d[-1]
            rows.append(dict(segment_index=offset+si,task_index=ti,task=tasks[ti],original_run_index=int(a[3]),
               stroke_index=down.get(int(a[3]),-1),pen_state=int(a[4]),raw_start=int(a[0]),raw_end_exclusive=int(b[1]),time_start=float(a[5]),time_last=float(b[6])))
        offset+=int(inv[:,2].max())+1
    return pd.DataFrame(rows)

def occluded_segments(model,batch,segments,present,sets):
    result=[]
    with torch.no_grad():
        for a in range(0,len(sets),32):
            mask=present[None].expand(min(32,len(sets)-a),-1).clone()
            for j,indices in enumerate(sets[a:a+32]):mask[j,torch.as_tensor(indices,device=DEV)]=False
            result.extend(model.classify_segments(batch,segments,mask).cpu().numpy())
    return np.asarray(result)

def coarse_blocks(person):
    result=[];pt=person['patch_task']
    for ti in range(14):
        at=np.flatnonzero(pt==ti)
        for j,indices in enumerate(np.array_split(at,min(8,len(at)))):
            result.append((ti,j,indices))
    return result

def run_fold(f):
    destination=OUT/'xai'/f'fold_{f}';destination.mkdir(parents=True,exist_ok=True)
    signature={name:sha(ROOT/name)for name in ('scripts/analyze_raw_temporal_xai.py','src/diagramo/temporal/explain.py')};signature['device']=DEV
    if(destination/'code.json').exists():assert json.loads((destination/'code.json').read_text())==signature,'XAI implementation changed; archive prior XAI outputs before rerunning'
    else:write(destination/'code.json',signature)
    if (destination/'complete.json').exists():print('REUSE XAI',f,flush=True);return
    ids,y,schema,assignment,splits=cohort();tr,te=splits[f]
    tasks=[];segments_rows=[];runs_rows=[];temporal=[];perturbations=[];embeddings=[];attention_rows=[];latency=[]
    for family in CONFIG['families']:
        for seed in CONFIG['seeds']:
            task_vectors=[];person_vectors=[]
            path=OUT/'folds'/f'fold_{f}'/'outer'/f'{family}_seed_{seed}.json';info=json.loads(path.read_text());assert sha(ROOT/info['checkpoint'])==info['checkpoint_sha256']
            model,bundle=load_checkpoint(ROOT/info['checkpoint']);norm=bundle['normalizer']
            saved=pd.read_csv(path.with_suffix('.oof.csv'),float_precision='round_trip').set_index('participant_id')
            for index in te:
                pid=ids[index];person=load_person(str(OUT/'cache'),pid);batch=tensor_person(pid,norm,family);representation=[];task_representation=[]
                # Warm the exact ragged shapes before attaching representation hooks.
                with torch.no_grad():model(batch,return_details=True)
                if DEV=='cuda':torch.cuda.synchronize()
                hook=model.head[0].register_forward_pre_hook(lambda module,inputs:representation.append(inputs[0].detach().cpu().numpy()))
                task_hook=model.task_ff.register_forward_hook(lambda module,inputs,output:task_representation.append(output.detach().cpu().numpy().reshape(14,32)))
                with torch.no_grad():
                    start=time.perf_counter();logit,details=model(batch,return_details=True);float(logit);seconds=time.perf_counter()-start
                    local=model.local(batch['x'],batch['length'])
                hook.remove();task_hook.remove();task_vectors.append(task_representation[0]);person_vectors.append(representation[0].reshape(32));logit=float(logit);p=float(expit(logit));np.testing.assert_allclose(p,saved.loc[pid,'probability'],atol=2e-6,rtol=0)
                latency.append(dict(model=family,seed=seed,fold=f,participant_id=pid,seconds=seconds,patches=len(person['length']),device=DEV,scope='warm '+DEV+' forward from normalized temporal tensors; excludes raw preparation and tensor transfer'))
                embeddings.append(dict(model=family,seed=seed,fold=f,participant_id=pid,**{'z'+str(k):float(v)for k,v in enumerate(representation[0].reshape(-1))}))
                masks=torch.ones((14,14),device=DEV,dtype=torch.bool);masks.fill_diagonal_(False)
                with torch.no_grad():
                    if model.hier:
                        seg,present=model(batch,local_tokens=local,return_segments=True)
                        removed=model.classify_segments(batch,seg,present[None].expand(14,-1),task_present=masks).cpu().numpy()
                    else:removed=np.array([float(model(batch,local_tokens=local,task_present=mask))for mask in masks])
                for ti,task in enumerate(schema['tasks']):tasks.append(dict(model=family,seed=seed,fold=f,participant_id=pid,label=int(y[index]),task=task,logit=logit,probability=p,removed_logit=float(removed[ti]),removed_probability=float(expit(removed[ti])),logit_delta=logit-float(removed[ti]),probability_delta=p-float(expit(removed[ti]))))
                if family!='hierarchical'or seed!=42:continue
                mapping=segment_mapping(person,schema['tasks']);S=len(mapping)
                per_segment=occluded_segments(model,batch,seg,present,[np.array([j])for j in range(S)])
                for row,value in zip(mapping.to_dict('records'),per_segment):segments_rows.append(dict(participant_id=pid,fold=f,**row,logit=logit,removed_logit=float(value),logit_delta=logit-float(value),probability_delta=p-float(expit(value))))
                runsets=[];runmeta=[]
                for(task,run),d in mapping.groupby(['task','original_run_index'],sort=False):
                    runsets.append(d.segment_index.to_numpy());row=d.iloc[0].to_dict();row.update(raw_end_exclusive=int(d.iloc[-1].raw_end_exclusive),time_last=float(d.iloc[-1].time_last),segments=len(d));runmeta.append(row)
                per_run=occluded_segments(model,batch,seg,present,runsets)
                for row,value in zip(runmeta,per_run):runs_rows.append(dict(participant_id=pid,fold=f,**row,logit=logit,removed_logit=float(value),logit_delta=logit-float(value),probability_delta=p-float(expit(value))))
                with torch.no_grad():
                    for ti,j,indices in coarse_blocks(person):
                        mask=torch.ones(len(local),device=DEV,dtype=torch.bool);mask[torch.as_tensor(indices,device=DEV)]=False
                        removed_logit=float(model(batch,patch_present=mask,local_tokens=local))
                        at=person['inventories'][ti];local_indices=indices-np.flatnonzero(person['patch_task']==ti)[0]
                        first,last=at[local_indices[0]],at[local_indices[-1]]
                        temporal.append(dict(participant_id=pid,fold=f,task=schema['tasks'][ti],task_index=ti,block=j,patch_start=int(indices[0]),patch_end_exclusive=int(indices[-1]+1),raw_start=int(first[0]),raw_end_exclusive=int(last[1]),time_start=float(first[5]),time_last=float(last[6]),logit=logit,removed_logit=removed_logit,logit_delta=logit-removed_logit,probability_delta=p-float(expit(removed_logit))))
                    # Complete pen-state runs are reordered, preserving their internal segments.
                    for kind in ('reverse_runs','shuffle_runs'):
                        for repeat in range(3):
                            generator=np.random.default_rng(np.random.SeedSequence([42,f,int(index),repeat,17 if kind=='reverse_runs'else 19]));order=[]
                            for ti in range(14):
                                d=mapping[mapping.task_index==ti];sets=[q.segment_index.to_numpy()for _,q in d.groupby('original_run_index',sort=False)]
                                seq=list(reversed(sets))if kind=='reverse_runs'else[sets[j]for j in generator.permutation(len(sets))]
                                order.extend(np.concatenate(seq))
                            changed=float(model.classify_segments(batch,seg,present,segment_order=torch.tensor(order,device=DEV)))
                            perturbations.append(dict(model=family,participant_id=pid,fold=f,label=int(y[index]),kind=kind,repeat=repeat,probability=float(expit(changed)),clean_probability=p,physical_realism='OOD: internal dynamics retained; global clocks/segment slot positions may conflict'))
                    for fraction in (.1,.25):
                        for repeat in range(3):
                            generator=np.random.default_rng(np.random.SeedSequence([42,f,int(index),repeat,int(fraction*100)]));chosen=generator.choice(S,max(1,int(S*fraction)),replace=False)
                            mask=present.clone();mask[torch.tensor(chosen,device=DEV)]=False;changed=float(model.classify_segments(batch,seg,mask))
                            perturbations.append(dict(model=family,participant_id=pid,fold=f,label=int(y[index]),kind='missing_segments_'+str(fraction),repeat=repeat,probability=float(expit(changed)),clean_probability=p,physical_realism='missing segment tokens; no zero-valued artificial movement'))
                if 'task_attention'in details:
                    a=details['task_attention'].cpu().numpy().mean((0,1))
                    for i in range(14):
                        for j in range(14):attention_rows.append(dict(participant_id=pid,fold=f,query_task=schema['tasks'][i],key_task=schema['tasks'][j],attention=float(a[i,j])))
            np.savez_compressed(destination/f'{family}_seed_{seed}_representations.npz',ids=ids[te],tasks=np.array(schema['tasks']),z_task=np.asarray(task_vectors),z_person=np.asarray(person_vectors))
            print('XAI',f,family,seed,'DONE',flush=True)
    for name,rows in [('task_occlusion',tasks),('segment_occlusion',segments_rows),('stroke_occlusion',runs_rows),('temporal_occlusion',temporal),('perturbation_oof',perturbations),('embeddings',embeddings),('attention_supplementary',attention_rows),('latency',latency)]:pd.DataFrame(rows).to_csv(destination/(name+'.csv'),index=False)
    write(destination/'complete.json',dict(complete=True,fold=f,device=DEV,segment_occlusions=len(segments_rows),run_occlusions=len(runs_rows),temporal_occlusions=len(temporal)))

def examples():
    ids,y,schema,assignment,splits=cohort();frames=[]
    for f in range(6):frames.append(pd.read_csv(OUT/'folds'/f'fold_{f}/outer/hierarchical_seed_42.oof.csv'))
    all_oof=pd.concat(frames);all_oof['category']=np.select([(all_oof.label==1)&(all_oof.prediction==1),(all_oof.label==0)&(all_oof.prediction==0),(all_oof.label==0)&(all_oof.prediction==1)],['TP','TN','FP'],default='FN')
    selected=all_oof.sort_values('participant_id').groupby('category',sort=False).head(1).copy();selected['selection']='lexicographically first ID per seed42 hierarchical OOF category';dest=OUT/'xai/examples';dest.mkdir(parents=True,exist_ok=True);selected.to_csv(dest/'selected_cases.csv',index=False)
    diagnostics=[];faith=[];deletion=[]
    for row in selected.itertuples():
        for seed in CONFIG['seeds']:
            info=json.loads((OUT/'folds'/f'fold_{row.fold}'/'outer'/f'hierarchical_seed_{seed}.json').read_text());model,b=load_checkpoint(ROOT/info['checkpoint']);batch=tensor_person(row.participant_id,b['normalizer'],'hierarchical')
            person=load_person(str(OUT/'cache'),row.participant_id)
            for baseline in ('zero_joint','structure_preserving'):
                result=converged_ig(model,batch,baseline);name=f'{row.participant_id}_seed_{seed}_{baseline}';np.savez_compressed(dest/(name+'.npz'),**result,task_mass=task_mass(result,batch))
                diagnostics.append(dict(participant_id=row.participant_id,category=row.category,fold=row.fold,seed=seed,baseline=baseline,kind='trained',residual=result['residual'],steps=result['steps'],logit=result['logit'],reference=result['reference']))
                if seed==42:
                    temporal=pd.read_csv(OUT/'xai'/f'fold_{row.fold}'/'temporal_occlusion.csv');temporal=temporal[temporal.participant_id==row.participant_id]
                    mass=np.abs(result['x']).sum((1,2))+np.abs(result['patch_meta']).sum(1)
                    blockmass=np.array([mass[int(q.patch_start):int(q.patch_end_exclusive)].sum()for q in temporal.itertuples()])
                    tasktab=pd.read_csv(OUT/'xai'/f'fold_{row.fold}'/'task_occlusion.csv');tasktab=tasktab[(tasktab.participant_id==row.participant_id)&(tasktab.model=='hierarchical')&(tasktab.seed==42)].set_index('task').loc[schema['tasks']]
                    faith.append(dict(participant_id=row.participant_id,baseline=baseline,temporal_spearman=float(spearmanr(blockmass,temporal.logit_delta.abs()).statistic),task_spearman=float(spearmanr(task_mass(result,batch),tasktab.logit_delta.abs()).statistic)))
                    with torch.no_grad():
                        local=model.local(batch['x'],batch['length']);clean=float(model(batch,local_tokens=local));sign=1 if clean>=0 else-1
                        for fraction in (.1,.25):
                            n=max(1,int(len(mass)*fraction))
                            for kind,indices in [('IG_top',np.argsort(-mass)[:n]),('IG_bottom',np.argsort(mass)[:n])]:
                                mask=torch.ones(len(mass),device=DEV,dtype=torch.bool);mask[torch.tensor(indices.copy(),device=DEV)]=False;changed=float(model(batch,patch_present=mask,local_tokens=local))
                                deletion.append(dict(participant_id=row.participant_id,baseline=baseline,fraction=fraction,kind=kind,confidence_logit_drop=sign*(clean-changed),probability=float(expit(changed))))
            if seed==42:
                for kind in ('random_head','random_all'):
                    randomized=copy.deepcopy(model);seed_all(9042)
                    for layer in (randomized.head.modules()if kind=='random_head'else randomized.modules()):
                        if hasattr(layer,'reset_parameters'):layer.reset_parameters()
                    result=converged_ig(randomized,batch,'zero_joint');np.savez_compressed(dest/(row.participant_id+'_'+kind+'.npz'),**result,task_mass=task_mass(result,batch))
                    diagnostics.append(dict(participant_id=row.participant_id,category=row.category,fold=row.fold,seed=9042,baseline='zero_joint',kind=kind,residual=result['residual'],steps=result['steps'],logit=result['logit'],reference=result['reference']))
            print('IG EXAMPLE',row.participant_id,seed,'DONE',flush=True)
    pd.DataFrame(diagnostics).to_csv(dest/'ig_validation.csv',index=False);pd.DataFrame(faith).to_csv(dest/'faithfulness.csv',index=False);pd.DataFrame(deletion).to_csv(dest/'deletion_curves.csv',index=False)
    write(dest/'complete.json',dict(complete=True,cases=len(selected),ig_artifacts=len(diagnostics)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folds',nargs='+',type=int,default=list(range(6)));parser.add_argument('--examples-only',action='store_true');parser.add_argument('--device',choices=['cpu','cuda'],default=DEV);args=parser.parse_args()
    if args.device=='cpu':
        import train_raw_temporal as engine
        engine.DEV='cpu';DEV='cpu';torch.set_num_threads(2)
    if args.examples_only:examples()
    else:
        for f in args.folds:run_fold(f)
