"""Independent data, Train-only fitting, checkpoint, temporal XAI and delivery audits."""
import argparse,itertools,subprocess
import torch
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from raw_temporal_common import *
from diagramo.temporal.preprocessing import preprocess
from diagramo.temporal.segmentation import windows
from diagramo.temporal.dataset import TrainNormalizer,load_person,memory_bank

def data_validation():
    protected=protect();ids,y,schema,assignment,splits=cohort();cleanup=json.loads((OUT/'cleanup.json').read_text())
    assert all(not(ROOT/p).exists()for p in cleanup['obsolete_tracked_files'])
    assert not(ROOT/'results/task_transformer').exists()
    tracked=set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'))
    assert not set(cleanup['obsolete_tracked_files'])&tracked
    rows=json.loads((OUT/'cache/manifest.json').read_text());stats=np.load(OUT/'cache/statistics.npz');np.testing.assert_array_equal(stats['ids'],ids)
    packed,offsets=memory_bank(str(OUT/'cache'));count=0;records=0;duplicate=0;gaps=0;maximum=0.
    for i,(pid,row)in enumerate(zip(ids,rows)):
        assert row['participant_id']==pid;recordings,meta,path=recording_map(pid,schema['tasks']);assert meta['group']==int(y[i]);assert sha(path)==row['source_sha256']
        for task in schema['tasks']:assert sha(raw_svc_path(pid,recordings[task][0]))==row['raw_source_sha256'][task]
        cachefile=OUT/'cache'/(pid+'.npz');assert sha(cachefile)==row['cache_sha256'];total=np.zeros((4,21));npoints=0;position=offsets[pid][0]
        with np.load(cachefile)as cached:
            for ti,task in enumerate(schema['tasks']):
                pre=preprocess(recordings[task][1]);inventory=windows(pre,128);np.testing.assert_array_equal(inventory,cached[f'm{ti}'])
                n=cached[f'n{ti}'];np.testing.assert_array_equal(n,inventory[:,1]-inventory[:,0]);mask=np.arange(128)[None,:]<n[:,None]
                for j,key in enumerate(('x','i')):
                    X=cached[f'{key}{ti}'];A=X[mask];reference=pre['values'if key=='x'else'index_values']
                    np.testing.assert_array_equal(A,reference);assert not np.any(X[~mask])
                    np.testing.assert_array_equal(X.astype('float16'),packed[j,position:position+len(X)])
                    maximum=max(maximum,float(np.max(np.abs(X-X.astype('float16').astype('float32')))))
                    total[2*j]+=A.sum(0,dtype='float64');total[2*j+1]+=np.square(A.astype('float64')).sum(0)
                for q,(a,b,seg,run,state,t0,t1)in enumerate(inventory):
                    a,b=int(a),int(b);assert (pre['segment'][a:b]==seg).all();assert (pre['run'][a:b]==run).all();assert(pre['pen'][a:b]==state).all();assert pre['time'][a]==t0 and pre['time'][b-1]==t1
                assert (pre['physical'][~pre['velocity_valid'],8:11]==0).all();assert(pre['physical'][~pre['acceleration_valid'],11:13]==0).all()
                assert np.isfinite(pre['values']).all();count+=len(pre['time']);records+=1;npoints+=len(pre['time']);position+=len(inventory)
                duplicate+=int((np.diff(pre['time'])==0).sum());gaps+=int((np.diff(pre['time'])>1).sum())
        assert position==offsets[pid][1];np.testing.assert_allclose(total/npoints,stats['statistics'][i],rtol=1e-12,atol=1e-10)
        if i%30==0:print('DATA VALIDATED',i+1,flush=True)
    expected=json.loads((OUT/'audit/summary.json').read_text());svc=pd.read_csv(OUT/'audit/raw_svc_comparison.csv')
    assert count==int(svc.raw_samples.sum())==expected['samples'] and records==3612 and duplicate==expected['duplicate_dt'] and gaps==expected['large_gaps']
    packinfo=json.loads((OUT/'cache/packed.json').read_text());assert maximum==packinfo['maximum_absolute_log_channel_quantization_error'];assert sha(OUT/'cache/packed.npy')==packinfo['sha256']
    write(OUT/'data_validation.json',dict(complete=True,protected_files=protected,participants=258,recordings=records,raw_samples=count,
      exact_sample_coverage=True,all_cached_channels_match_raw_preprocessing=True,all_timestamp_and_stroke_alignment_checked=True,derivative_masks_checked=True,
      original_float32_statistics_recomputed=True,packed_storage_exactly_validated=True,maximum_storage_error=maximum,obsolete_tracked_task2_removed=True))
    print('DATA VALIDATION COMPLETE',flush=True)

def fits_validation(fold_subset=None,device=None):
    if device is not None:
        import train_raw_temporal as engine
        engine.DEV=device
        if device=='cpu':torch.set_num_threads(2)
    from train_raw_temporal import CONFIG,load_checkpoint,probability,signature,statistics,seed_all
    seed_all(42)
    sig=signature();ids,y,schema,assignment,splits=cohort();stats=statistics();inner_count=outer_count=0
    requested=set(range(6))if fold_subset is None else set(fold_subset)
    assert requested and requested<=set(range(6))
    for f,(tr,te)in enumerate(splits):
        if f not in requested:continue
        folder=OUT/'folds'/f'fold_{f}';assert json.loads((folder/'complete.json').read_text())['complete']
        inner=list(StratifiedKFold(3,shuffle=True,random_state=42).split(tr,y[tr]));search=[]
        for scope in ('inner','outer'):
            for p in sorted((folder/scope).glob('*.json')):
                if p.name.endswith('_history.json'):continue
                info=json.loads(p.read_text());assert info['signature']==sig
                if scope=='inner':
                    a,b=inner[info['inner']];fit_idx,evaluation=tr[a],tr[b];assert not set(ids[te])&set(info['evaluation_ids']);inner_count+=1
                else:fit_idx,evaluation=tr,te;outer_count+=1
                assert info['fit_ids']==ids[fit_idx].tolist()and info['evaluation_ids']==ids[evaluation].tolist();assert not set(info['fit_ids'])&set(info['evaluation_ids'])
                assert sha(ROOT/info['checkpoint'])==info['checkpoint_sha256'];model,bundle=load_checkpoint(ROOT/info['checkpoint']);norm=bundle['normalizer']
                assert norm.fit_ids==info['fit_ids'];fresh=TrainNormalizer().fit(stats,fit_idx,ids[fit_idx],info['family']=='index_time')
                np.testing.assert_array_equal(norm.mean,fresh.mean);np.testing.assert_array_equal(norm.scale,fresh.scale)
                assert sum(a.numel()for a in model.parameters())==info['parameters']<50000
                pred,_=probability(model,norm,ids[evaluation],info['family']);saved=pd.read_csv(p.with_suffix('.oof.csv'),float_precision='round_trip')
                np.testing.assert_array_equal(saved.participant_id,ids[evaluation]);np.testing.assert_array_equal(saved.label,y[evaluation]);np.testing.assert_allclose(pred,saved.probability,atol=2e-6,rtol=0)
                np.testing.assert_array_equal(saved.prediction,(saved.probability>.5).astype(int))
                for k,v in metrics(saved.label,saved.probability).items():np.testing.assert_allclose(v,info[k],atol=1e-12,rtol=0)
                history=json.loads(p.with_name(p.stem+'_history.json').read_text());assert info['epochs_ran']==len(history)
                if scope=='inner':
                    best=next(r for r in history if r['epoch']==info['best_epoch']);assert best['validation_loss']<=min(r['validation_loss']for r in history)+1e-4;search.append(info)
                else:assert info['epochs_ran']==info['epochs']==info['best_epoch']
            print('FIT CHECKPOINTS VALIDATED',f,scope,flush=True)
        main=pd.DataFrame([r for r in search if r['family']in CONFIG['families']]);means=main.groupby(['family','candidate']).agg(f1=('f1_macro','mean'),auc=('auc','mean'),epochs=('best_epoch','median')).reset_index();decision=json.loads((folder/'selection.json').read_text())
        winner=means.sort_values(['f1','auc','candidate','family'],ascending=[False,False,True,True]).iloc[0];assert decision['nested_winner']==winner.family and decision['candidate']==winner.candidate
        for family in CONFIG['families']:
            best=means[means.family==family].sort_values(['f1','auc','candidate'],ascending=[False,False,True]).iloc[0]
            assert decision['choices'][family]['candidate']==best.candidate and decision['choices'][family]['epochs']==max(1,int(best.epochs))
        for family in CONFIG['ablations']:
            rows=[r for r in search if r['family']==family];assert len(rows)==3
            assert decision['choices'][family]['candidate']==decision['choices']['hierarchical']['candidate'];assert decision['choices'][family]['epochs']==max(1,int(np.median([r['best_epoch']for r in rows])))
        for p in (folder/'outer').glob('*.json'):
            if p.name.endswith('_history.json'):continue
            info=json.loads(p.read_text());assert info['candidate']==decision['choices'][info['family']]['candidate'];assert info['epochs']==decision['choices'][info['family']]['epochs']
    assert inner_count==36*len(requested) and outer_count==16*len(requested)
    if requested!=set(range(6)):
        write(OUT/'partial_validation'/('folds_'+'_'.join(map(str,sorted(requested)))+'.json'),
            dict(complete=False,scope='Partial checkpoint check only; consult validation.json for final whole-study status',checks_passed=True,
                folds=sorted(requested),inner_fits=inner_count,outer_fits=outer_count,all_requested_predictions_replayed=True,
                train_only_normalization_verified=True,inner_only_selection_verified=True,training_signature=sig,device=device))
        return
    oof=pd.read_csv(OUT/'tables/oof_predictions.csv',float_precision='round_trip');summary=pd.read_csv(OUT/'tables/performance_summary.csv')
    for(model,seed),d in oof.groupby(['model','seed']):
        assert len(d)==258 and d.participant_id.is_unique;row=summary[(summary.model==model)&(summary.seed==seed)].iloc[0]
        for k,v in metrics(d.label,d.probability).items():np.testing.assert_allclose(v,row[k],atol=1e-12,rtol=0)
    base=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv',float_precision='round_trip');reference=oof[oof.model=='kinematic_lightgbm_reference'].set_index('participant_id').loc[ids]
    np.testing.assert_array_equal(reference.prediction,base.prediction);np.testing.assert_allclose(reference.probability,base.probability,atol=1e-15,rtol=0)
    write(OUT/'fit_validation.json',dict(complete=True,inner_fits=inner_count,outer_fits=outer_count,normalization_train_only=True,selection_inner_only=True,all_checkpoint_predictions_replayed=True,baseline_oof_unchanged=True,all_participant_metrics_checked=True))

def xai_validation():
    # XAI artifacts, including randomized weights, were produced on CPU.
    # CPU and CUDA RNG streams differ even when given the same seed.
    import train_raw_temporal as engine
    engine.DEV='cpu';torch.set_num_threads(2)
    from train_raw_temporal import load_checkpoint,tensor_person,DEV,seed_all
    from analyze_raw_temporal_xai import segment_mapping,coarse_blocks
    ids,y,schema,assignment,splits=cohort();aligned=0;replayed=0;sigmoid_roundoff=0.
    expected_code={name:sha(ROOT/name)for name in ('scripts/analyze_raw_temporal_xai.py','src/diagramo/temporal/explain.py')};expected_code['device']=DEV
    for f,(tr,te)in enumerate(splits):
        d=OUT/'xai'/f'fold_{f}';assert json.loads((d/'complete.json').read_text())['complete']
        assert json.loads((d/'code.json').read_text())==expected_code
        tasks=pd.read_csv(d/'task_occlusion.csv',float_precision='round_trip');segcsv=pd.read_csv(d/'segment_occlusion.csv',float_precision='round_trip');runscsv=pd.read_csv(d/'stroke_occlusion.csv',float_precision='round_trip');temporals=pd.read_csv(d/'temporal_occlusion.csv',float_precision='round_trip')
        assert len(tasks)==43*4*3*14 and not tasks.duplicated(['model','seed','participant_id','task']).any()
        latency=pd.read_csv(d/'latency.csv');assert len(latency)==43*4*3
        assert (latency.device==DEV).all() and np.isfinite(latency.seconds).all() and (latency.seconds>0).all()
        assert not latency.duplicated(['model','seed','participant_id']).any()
        for(family,seed),group in tasks.groupby(['model','seed']):
            saved=pd.read_csv(OUT/'folds'/f'fold_{f}'/'outer'/f'{family}_seed_{seed}.oof.csv',float_precision='round_trip').set_index('participant_id')
            np.testing.assert_allclose(group.probability,group.participant_id.map(saved.probability),atol=2e-6,rtol=0)
            np.testing.assert_array_equal(group.label,group.participant_id.map(saved.label))
            assert set(group.participant_id)==set(ids[te]) and set(group.task)==set(schema['tasks'])
            with np.load(d/f'{family}_seed_{seed}_representations.npz')as representation:
                np.testing.assert_array_equal(representation['ids'],ids[te]);np.testing.assert_array_equal(representation['tasks'],schema['tasks'])
                assert representation['z_task'].shape==(43,14,32) and representation['z_person'].shape==(43,32)
                assert np.isfinite(representation['z_task']).all() and np.isfinite(representation['z_person']).all()
        for seed in (42,43,44):
            info=json.loads((OUT/'folds'/f'fold_{f}'/'outer'/f'hierarchical_seed_{seed}.json').read_text());model,b=load_checkpoint(ROOT/info['checkpoint'])
            for i in te:
                pid=ids[i];person=load_person(str(OUT/'cache'),pid);batch=tensor_person(pid,b['normalizer'],'hierarchical')
                with torch.no_grad():
                    seg,present=model(batch,return_segments=True);clean=float(model.classify_segments(batch,seg,present));mask=torch.ones((14,14),device=DEV,dtype=torch.bool);mask.fill_diagonal_(False)
                    removed=model.classify_segments(batch,seg,present[None].expand(14,-1),task_present=mask).cpu().numpy()
                s=tasks[(tasks.participant_id==pid)&(tasks.model=='hierarchical')&(tasks.seed==seed)].set_index('task').loc[schema['tasks']]
                np.testing.assert_allclose(removed,s.removed_logit,atol=3e-6,rtol=0);np.testing.assert_allclose(clean-s.removed_logit,s.logit_delta,atol=3e-6,rtol=0);replayed+=14
                if seed!=42:continue
                mapping=segment_mapping(person,schema['tasks']);saved=segcsv[segcsv.participant_id==pid].reset_index(drop=True)
                for col in mapping:np.testing.assert_array_equal(mapping[col],saved[col])
                aligned+=len(mapping)
                # Independent deterministic replay of first/last segment of every task;
                # all rows have full alignment and arithmetic checks, not claimed as all replayed.
                chosen=np.unique(mapping.groupby('task',sort=False).segment_index.agg(['first','last']).to_numpy().reshape(-1));masks=present[None].expand(len(chosen),-1).clone();masks[torch.arange(len(chosen),device=DEV),torch.tensor(chosen,device=DEV)]=False
                with torch.no_grad():values=model.classify_segments(batch,seg,masks).cpu().numpy()
                np.testing.assert_allclose(values,saved.iloc[chosen].removed_logit,atol=3e-6,rtol=0);replayed+=len(chosen)
                for row in temporals[temporals.participant_id==pid].itertuples():
                    inv=person['inventories'][row.task_index];start=np.flatnonzero(person['patch_task']==row.task_index)[0];a=row.patch_start-start;bidx=row.patch_end_exclusive-start-1
                    assert inv[a,0]==row.raw_start and inv[bidx,1]==row.raw_end_exclusive and inv[a,5]==row.time_start and inv[bidx,6]==row.time_last
                for row in runscsv[runscsv.participant_id==pid].itertuples():
                    actual=mapping[(mapping.task==row.task)&(mapping.original_run_index==row.original_run_index)];assert len(actual)==row.segments and actual.iloc[0].raw_start==row.raw_start and actual.iloc[-1].raw_end_exclusive==row.raw_end_exclusive
        for frame in (segcsv,runscsv,temporals):
            np.testing.assert_allclose(frame.logit-frame.removed_logit,frame.logit_delta,atol=1e-12,rtol=0)
            # Segment/run logits came from NumPy float32 tensor arrays. The
            # exporter applied expit BEFORE converting that scalar to float.
            # Temporal logits were converted to Python float BEFORE expit.
            removed=frame.removed_logit.to_numpy(dtype='float64' if frame is temporals else 'float32')
            expected=expit(frame.logit.to_numpy())-expit(removed).astype('float64')
            np.testing.assert_allclose(expected,frame.probability_delta,atol=1e-12,rtol=0)
            error=float(np.max(np.abs((expit(frame.logit)-expit(frame.removed_logit))-frame.probability_delta)))
            assert error<=np.finfo('float32').eps
            sigmoid_roundoff=max(sigmoid_roundoff,error)
        print('XAI ALIGNMENT AND REPLAY VALIDATED',f,flush=True)
    cases=pd.read_csv(OUT/'xai/examples/selected_cases.csv');igcount=0
    for row in cases.itertuples():
        for p in (OUT/'xai/examples').glob(row.participant_id+'*.npz'):
            with np.load(p)as a:
                import copy
                from diagramo.temporal.explain import references,task_mass
                seed=42 if '_random_'in p.name else int(p.name.split('_seed_')[1].split('_')[0])
                info=json.loads((OUT/'folds'/f'fold_{row.fold}'/'outer'/f'hierarchical_seed_{seed}.json').read_text());model,bundle=load_checkpoint(ROOT/info['checkpoint']);batch=tensor_person(row.participant_id,bundle['normalizer'],'hierarchical')
                if '_random_'in p.name:
                    seed_all(9042)
                    for layer in (model.head.modules()if'random_head'in p.name else model.modules()):
                        if hasattr(layer,'reset_parameters'):layer.reset_parameters()
                baseline=references(batch,str(a['baseline_kind']))
                with torch.no_grad():
                    logit=float(model(batch));ref=float(model(dict(batch,**baseline)))
                np.testing.assert_allclose(logit,a['logit'],atol=3e-6,rtol=0);np.testing.assert_allclose(ref,a['reference'],atol=3e-6,rtol=0)
                residual=a['x'].sum()+a['patch_meta'].sum()+a['segment_meta'].sum()-(float(a['logit'])-float(a['reference']))
                assert abs(residual)<=.003+.003*abs(float(a['logit'])-float(a['reference']));np.testing.assert_allclose(residual,a['residual'],atol=1e-10,rtol=0)
                assert a['x'].ndim==3 and a['x'].shape[1]==21
                np.testing.assert_allclose(task_mass(a,batch),a['task_mass'],atol=1e-10,rtol=0)
                mask=np.arange(a['x'].shape[-1])[None,:]>=batch['length'].cpu().numpy()[:,None]
                assert not np.any(a['x'].transpose(0,2,1)[mask]);igcount+=1
    write(OUT/'xai_validation.json',dict(complete=True,all_segment_rows_raw_aligned=aligned,task_and_sampled_segment_predictions_replayed=replayed,
       full_stroke_and_temporal_alignment_checked=True,ig_artifacts=igcount,all_ig_completeness_checked=True,device=DEV,
       probability_delta_dtype_arithmetic_reproduced=True,maximum_float32_sigmoid_roundoff=sigmoid_roundoff,
       note='Every segment/stroke/time row checked for mapping/arithmetic; model replay uses all hierarchical task occlusions and first/last segment per task. Randomization and baseline changes are diagnostics, not proof of clinical relevance.'))

def final():
    import nbformat
    from PIL import Image
    from train_raw_temporal import signature
    protect()
    signature()  # Reject changes to frozen training code/configuration.
    cleanup=json.loads((OUT/'cleanup.json').read_text())
    assert all(not(ROOT/p).exists()for p in cleanup['obsolete_tracked_files']+cleanup.get('obsolete_ignored_helpers_removed',[]))
    sources=json.loads((OUT/'audit/source_hashes.json').read_text())
    for source in sources:assert sha(ROOT/source['path'])==source['sha256'],source['path']
    for name in ('data_validation','fit_validation','xai_validation'):assert json.loads((OUT/(name+'.json')).read_text())['complete']
    notebook_check=json.loads((OUT/'notebook_validation.json').read_text())
    assert len(notebook_check)==1 and notebook_check[0]['notebook']=='06_raw_temporal_graphomotor.ipynb'
    assert notebook_check[0]['errors']==0 and notebook_check[0]['executed_code_cells']>0
    config=json.loads((ROOT/'configs/raw_temporal.json').read_text());manifest=json.loads((OUT/'figures/manifest.json').read_text());assert len(manifest)>=12
    review=json.loads((OUT/'visual_review.json').read_text());assert review['figures_reviewed']==len(manifest)
    for row in manifest:
        assert review['figure_png_sha256'][row['name']]==sha(OUT/'figures'/(row['name']+'.png'))
        for rel,digest in row['sources'].items():assert sha(ROOT/rel)==digest,rel
        for ext,digest in row['outputs'].items():
            p=OUT/'figures'/(row['name']+'.'+ext);assert sha(p)==digest
            if ext=='png':
                with Image.open(p)as im:assert min(im.info['dpi'])>=299
    n=nbformat.read(ROOT/'notebooks/06_raw_temporal_graphomotor.ipynb',as_version=4);nbformat.validate(n);code=[c for c in n.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in code);assert not[o for c in code for o in c.outputs if o.output_type=='error']
    for pattern in ('test_raw_temporal.py','test_kinematic_xai.py','test_baseline.py'):
        test=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p',pattern,'-v'],capture_output=True,text=True,encoding='utf-8');(OUT/(pattern+'.log')).write_text(test.stdout+test.stderr,encoding='utf-8');assert test.returncode==0,test.stderr
    source=[p for folder in ('src/diagramo/temporal','scripts','tests')for p in (ROOT/folder).glob('*.py')if folder.endswith('temporal')or'raw_temporal'in p.name]
    for p in source:compile(p.read_text(encoding='utf-8'),str(p),'exec')
    artifacts=[p for folder in ('audit','tables','xai','folds')for p in (OUT/folder).rglob('*')if p.is_file()]
    artifacts.extend(OUT/name for name in ('config.json','runtime.json','provenance.json','cpu_hardware.json','training_code.json','fold_assignments.csv',
        'data_validation.json','fit_validation.json','xai_validation.json','notebook_validation.json','cleanup.json','protected_manifest.json','baseline_metadata_preservation.json','visual_review.json','figures/manifest.json'))
    artifacts.extend(OUT.glob('xai_warmup_equivalence_fold_*.json'))
    write(OUT/'reproducibility_manifest.json',dict(complete=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p)for p in source},
        artifact_sha256={p.relative_to(ROOT).as_posix():sha(p)for p in artifacts},config_sha256=sha(ROOT/'configs/raw_temporal.json'),runtime_config_sha256=sha(ROOT/'configs/raw_temporal_runtime.json'),
        report_sha256=sha(OUT/'REPORT_VI.md'),notebook_sha256=sha(ROOT/'notebooks/06_raw_temporal_graphomotor.ipynb')))
    delivery=json.loads((OUT/'delivery_progress.json').read_text())if(OUT/'delivery_progress.json').exists()else{}
    delivery.update(stage='complete',complete=True,final_validation='passed');write(OUT/'delivery_progress.json',delivery)
    write(OUT/'validation.json',dict(complete=True,stage='complete',protected_files_unchanged=protect(),outer_folds=6,inner_folds=3,participants=258,
      recordings=3612,raw_samples_covered=json.loads((OUT/'data_validation.json').read_text())['raw_samples'],raw_source_files_rechecked=len(sources),neural_handcrafted_input=False,inner_fits=216,outer_fits=96,figures=len(manifest),
      raw_temporal_unit_tests=16,task1_unit_tests=8,baseline_regression_tests=6,notebook_executed=True,numeric_table_outputs=sum('text/html'in o.get('data',{})for c in code for o in c.outputs),
      old_task2_active_files_removed=True,old_task2_historical_commits_preserved=True,external_validation=False,cross_language=False,fusion=False,
      limitations=['Exploratory reuse of previously inspected participant folds','No calibrated sensor ground truth','Bounded context; unequal architectural parameter counts','Temporal perturbations can be out of distribution']))
    print('FINAL RAW TEMPORAL VALIDATION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['data','fits','xai','final'],default='final')
    p.add_argument('--folds',nargs='+',type=int,help='Optional incremental checkpoint checks; never marks whole study complete')
    p.add_argument('--device',choices=['cpu','cuda'],help='Optional device for checkpoint replay')
    args=p.parse_args()
    if args.phase=='fits':fits_validation(args.folds,args.device)
    else:
        assert args.folds is None and args.device is None,'--folds/--device apply to --phase fits only'
        {'data':data_validation,'xai':xai_validation,'final':final}[args.phase]()
