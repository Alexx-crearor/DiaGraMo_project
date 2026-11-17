"""Controlled development experiments; outer Test participants never loaded."""
import argparse
import copy
import time
from functools import lru_cache
from raw_temporal_v2_common import *
from diagramo.temporal.dataset import TrainNormalizer,load_person,collate_people
from diagramo.temporal_v2.models import RecoveryModel

CONFIG_PATH='configs/raw_temporal_v2/phase1.json'
CONFIG=json.loads((ROOT/CONFIG_PATH).read_text())
SOURCES=['scripts/train_raw_temporal_v2_phase1.py','src/diagramo/temporal_v2/models.py']


@lru_cache(maxsize=258)
def metadata(pid):
    with np.load(LOCAL/'physical_targets'/f'{pid}.npz') as data:
        return {key:data[key].copy() for key in ('token_meta','patch_duration','segment_duration','task_duration')}


def tensor(pid,normalizer,family):
    item=normalizer.transform(load_person(str(CACHE),str(pid)),DEVICE,family)
    item.update({key:torch.as_tensor(value,device=DEVICE) for key,value in metadata(str(pid)).items()})
    return item


def predict(model,ids,normalizer,capture=False):
    model.eval();logits=[];embeddings=[]
    with torch.no_grad():
        for start in range(0,len(ids),2):
            batch=collate_people([tensor(pid,normalizer,model.family) for pid in ids[start:start+2]])
            output,detail=model(batch,return_details=True)
            logits.extend(output.cpu().numpy());embeddings.extend(detail['person'].cpu().numpy())
    return np.asarray(logits,dtype='float64'),np.asarray(embeddings) if capture else None


def experiments():
    result=[]
    for pool in CONFIG['local_pooling']:
        result.append(dict(name='cnn_local_'+pool,family='cnn',local_pool=pool))
    result.append(dict(name='hierarchical_mil',family='hierarchical'))
    for stage in CONFIG['aggregation_stages']:
        for method in CONFIG['aggregation_methods']:
            result.append(dict(name=f'hierarchical_{stage}_{method}',family='hierarchical',
                aggregation_stage=stage,aggregation_method=method))
    ids,y,schema,fit,val,test=development()
    # Resolve actual protected schema instead of assuming task index ordering.
    names=schema['task_names'] if 'task_names' in schema else schema['tasks']
    if isinstance(names[0],dict):names=[x.get('name',x.get('task')) for x in names]
    for task in CONFIG['task_specific']:
        result.append(dict(name='task_'+task,family='cnn',task_indices=[names.index(task)]))
    result.extend([dict(name='cnn_original_budget20',family='cnn',epochs=20),
        dict(name='cnn_lr003',family='cnn',learning_rate=.003),
        dict(name='hierarchical_lr003',family='hierarchical',learning_rate=.003)])
    return result


def fit(spec):
    seed_all(42);ids,y,schema,train,val,test=development()
    experiment='p1_'+spec['name'];dest=OUT/'phase1_bottleneck'/experiment;dest.mkdir(parents=True,exist_ok=True)
    sig=signature(CONFIG_PATH,SOURCES)
    manifest=json.loads((OUT/'phase1_bottleneck/physical_targets_manifest.json').read_text())
    assert manifest['complete']
    result_path=dest/'result.json'
    if result_path.exists():
        saved=json.loads(result_path.read_text());assert saved['signature']==sig['signature']
        assert sha(ROOT/saved['checkpoint'])==saved['checkpoint_sha256'];print('REUSE',experiment,flush=True);return saved
    architecture={k:v for k,v in spec.items() if k not in ('name','epochs','learning_rate')}
    architecture['dropout']=CONFIG['dropout']
    model=RecoveryModel(**architecture).to(DEVICE)
    begin(experiment,'1',spec['family'],sig,hypothesis='H0 aggregation / H1 local compression; development only',
        input_representation='Original cached chronological21 channels; all source samples',
        ablation=spec,split='outer0 Train: inner0 fit143 validation72; Test43 excluded',seed=42)
    write(dest/'signature.json',sig)
    with np.load(CACHE/'statistics.npz') as data:statistics=data['statistics']
    normalizer=TrainNormalizer().fit(statistics,train,ids[train])
    assert not set(normalizer.fit_ids)&set(ids[val]) and not set(normalizer.fit_ids)&set(ids[test])
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec.get('learning_rate',CONFIG['learning_rate']),weight_decay=CONFIG['weight_decay'])
    checkpoint=LOCAL/'phase1'/f'{experiment}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    resume=checkpoint.with_suffix('.resume.pt');rng=np.random.default_rng(42)
    history=[];best=None;best_loss=float('inf');best_epoch=0;stale=0;start_epoch=1;elapsed=0.
    if resume.exists():
        state=torch.load(resume,map_location=DEVICE,weights_only=False);assert state['signature']==sig['signature']
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];best_loss=state['best_loss'];best_epoch=state['best_epoch']
        stale=state['stale'];start_epoch=state['epoch']+1;elapsed=state['elapsed']
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'].cpu())
        if DEVICE=='cuda':torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    maximum=spec.get('epochs',CONFIG['epochs']);sync();began=time.perf_counter()
    if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,maximum+1):
        model.train();order=rng.permutation(train);online=[];gradient=[]
        for pos in range(0,len(order),RESOURCE['effective_batch_participants']):
            group=order[pos:pos+RESOURCE['effective_batch_participants']];optimizer.zero_grad(set_to_none=True)
            for micro in range(0,len(group),RESOURCE['micro_batch_participants']):
                at=group[micro:micro+RESOURCE['micro_batch_participants']]
                batch=collate_people([tensor(ids[i],normalizer,model.family) for i in at])
                logit=model(batch);label=torch.tensor(y[at],device=DEVICE,dtype=torch.float32)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,label)
                assert torch.isfinite(loss);(loss*len(at)/len(group)).backward();online.extend([float(loss.detach())]*len(at))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip'])
            assert torch.isfinite(norm);gradient.append(float(norm));optimizer.step()
        logits,_=predict(model,ids[val],normalizer);p=1/(1+np.exp(-logits))
        bce=float(np.mean(np.logaddexp(0,logits)-y[val]*logits))
        history.append(dict(epoch=epoch,online_train_bce=float(np.mean(online)),validation_bce=bce,
            gradient_norm_mean=float(np.mean(gradient)),**metrics(y[val],p)))
        if bce<best_loss-1e-4:
            best_loss=bce;best_epoch=epoch;best=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        stopping=epoch>=CONFIG['minimum_epochs'] and stale>=CONFIG['patience']
        if epoch%10==0 or stopping or epoch==maximum:
            sync();duration=elapsed+time.perf_counter()-began
            torch.save(dict(signature=sig['signature'],model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch,history=history,best=best,best_loss=best_loss,best_epoch=best_epoch,stale=stale,
                elapsed=duration,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state() if DEVICE=='cuda' else None),resume)
            pd.DataFrame(history).to_csv(dest/'learning_curve.csv',index=False);resource_check()
            print('PHASE1',spec['name'],'epoch',epoch,'valBCE',round(bce,5),'F1',round(history[-1]['f1_macro'],4),flush=True)
        if stopping:break
    model.load_state_dict(best);logits,z=predict(model,ids[val],normalizer,True);p=1/(1+np.exp(-logits))
    train_logits,_=predict(model,ids[train],normalizer)
    pd.DataFrame(dict(participant_id=ids[val],label=y[val],logit=logits,probability=p)).to_csv(dest/'validation_predictions.csv',index=False)
    pd.DataFrame(dict(participant_id=ids[train],label=y[train],logit=train_logits,
        probability=1/(1+np.exp(-train_logits)))).to_csv(dest/'training_predictions.csv',index=False)
    np.savez_compressed(dest/'validation_embeddings.npz',ids=ids[val],z=z)
    sync();duration=elapsed+time.perf_counter()-began
    torch.save(dict(state={k:v.cpu() for k,v in best.items()},normalizer=normalizer,architecture=model.config,
        fit_ids=ids[train].tolist(),validation_ids=ids[val].tolist(),best_epoch=best_epoch,
        target_manifest_signature=manifest['signature'],signature=sig['signature']),checkpoint)
    result=dict(experiment_id=experiment,signature=sig['signature'],architecture=model.config,
        metrics=metrics(y[val],p),train_metrics=metrics(y[train],1/(1+np.exp(-train_logits))),
        best_epoch=best_epoch,epochs_run=epoch,training_seconds=duration,parameters=sum(x.numel() for x in model.parameters()),
        peak_gpu_tensor_mib=torch.cuda.max_memory_allocated()/1024**2 if DEVICE=='cuda' else 0,
        embedding_variance=float(z.var(0).mean()),checkpoint=str(checkpoint.relative_to(ROOT)).replace('\\','/'),
        checkpoint_sha256=sha(checkpoint),fit_ids=ids[train].tolist(),validation_ids=ids[val].tolist(),
        outer_test_evaluated=False,complete=True,scope='Exploratory development; not Nested CV')
    write(result_path,result)
    ledger(experiment,training_status='completed',completion_status='completed',
        validation_metrics=result['metrics'],best_checkpoint=result['checkpoint'],
        parameter_count=result['parameters'],gpu_peak_memory=result['peak_gpu_tensor_mib'],
        training_cost=dict(seconds=duration,epochs=epoch),artifact_paths=str(dest.relative_to(ROOT)),
        reason='Development fit; checkpoint replay pending')
    print('COMPLETE',experiment,result['metrics'],flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only');parser.add_argument('--list',action='store_true');args=parser.parse_args()
    specs=experiments()
    if args.list:print(json.dumps(specs,indent=2));raise SystemExit
    with gpu_worker():
        for spec in specs:
            if args.only and spec['name']!=args.only:continue
            fit(spec)
