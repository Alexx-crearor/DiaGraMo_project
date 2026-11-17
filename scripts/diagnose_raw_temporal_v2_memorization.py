"""Phase 0: current, unchanged encoders must optimize balanced tiny Train bags.

Dropout and weight decay are disabled only for these optimization diagnostics.
No held-out real participant is scored by this script.
"""
import argparse
import copy
import math
import time
from scipy.spatial.distance import pdist
from raw_temporal_v2_common import *
from diagramo.temporal.dataset import TrainNormalizer, load_person, collate_people
from diagramo.temporal.hierarchical_model import RawTemporalModel

CONFIG_PATH = 'configs/raw_temporal_v2/phase0.json'
CONFIG = json.loads((ROOT / CONFIG_PATH).read_text())
SCRIPT = 'scripts/diagnose_raw_temporal_v2_memorization.py'


def gradient_summary(model, before, epoch, step):
    grouped = {}
    for name, parameter in model.named_parameters():
        module = name.split('.')[0]
        row = grouped.setdefault(module, dict(epoch=epoch, step=step, module=module,
            gradient_squared_norm=0., update_squared_norm=0., parameter_squared_norm=0.,
            gradient_elements=0, gradient_zeros=0, gradient_nonfinite=0, unused_parameters=0))
        row['parameter_squared_norm'] += float(parameter.detach().square().sum())
        row['update_squared_norm'] += float((parameter.detach()-before[name]).square().sum())
        if parameter.grad is None:
            row['unused_parameters'] += parameter.numel()
        else:
            gradient = parameter.grad.detach()
            row['gradient_elements'] += gradient.numel()
            row['gradient_zeros'] += int((gradient == 0).sum())
            row['gradient_nonfinite'] += int((~torch.isfinite(gradient)).sum())
            row['gradient_squared_norm'] += float(gradient.square().sum())
    for row in grouped.values():
        row['gradient_norm'] = math.sqrt(row.pop('gradient_squared_norm'))
        row['parameter_update_norm'] = math.sqrt(row.pop('update_squared_norm'))
        row['parameter_norm'] = math.sqrt(row.pop('parameter_squared_norm'))
        row['gradient_zero_fraction'] = row['gradient_zeros']/max(row['gradient_elements'],1)
    return list(grouped.values())


def predictions(model, tensors, batch_size=2, capture=False):
    model.eval(); logits=[]; embeddings=[]
    handle = model.head[0].register_forward_pre_hook(
        lambda module, inputs: embeddings.append(inputs[0].detach().cpu().numpy())) if capture else None
    with torch.no_grad():
        for start in range(0,len(tensors),batch_size):
            logits.extend(model(collate_people(tensors[start:start+batch_size])).cpu().numpy())
    if handle: handle.remove()
    return np.asarray(logits, dtype='float64'), np.concatenate(embeddings) if capture else None


def activation_audit(model, tensors):
    observations=[];handles=[]
    def hook(name):
        def record(module, inputs, output):
            a=output[0] if isinstance(output,tuple) else output
            if isinstance(a,torch.Tensor):
                a=a.detach().float()
                observations.append(dict(module=name, elements=a.numel(), mean=float(a.mean()),
                    variance=float(a.var(unbiased=False)), zero_fraction=float((a==0).float().mean()),
                    nonfinite=int((~torch.isfinite(a)).sum())))
        return record
    for name in ('local','pool','stroke','task_context','tcn','person_context','task_ff','head'):
        module=getattr(model,name,None)
        if module is not None:handles.append(module.register_forward_hook(hook(name)))
    predictions(model,tensors)
    for handle in handles:handle.remove()
    return observations


def fit_tiny(family,size):
    seed_all(42); chosen=balanced_tiny(size)
    ids,y,_,_,_,excluded=development(); assert not set(chosen)&set(excluded)
    experiment=f'p0_tiny_{family}_{size}_seed42'
    destination=OUT/'phase0_diagnostics'/experiment;destination.mkdir(parents=True,exist_ok=True)
    sig=signature(CONFIG_PATH,[SCRIPT]); target=destination/'result.json'
    if target.exists():
        saved=json.loads(target.read_text());assert saved['signature']==sig['signature']
        assert sha(ROOT/saved['checkpoint'])==saved['checkpoint_sha256']
        print('REUSE',experiment,flush=True);return saved
    begin(experiment,'0',family,sig,hypothesis='Engineering optimization capacity; not generalization',
        input_representation='All original cached chronological samples, all14 tasks',
        ablation='dropout0_weight_decay0',split='balanced subset of development_fit only',seed=42)
    write(destination/'signature.json',sig)
    with np.load(CACHE/'statistics.npz') as source:statistics=source['statistics']
    normalizer=TrainNormalizer().fit(statistics,chosen,ids[chosen])
    tensors=[normalizer.transform(load_person(str(CACHE),pid),DEVICE,family) for pid in ids[chosen]]
    model=RawTemporalModel(family,d_model=32,dropout=0.).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['tiny_learning_rate'],weight_decay=0.)
    rng=np.random.default_rng(42);history=[];gradient_rows=[];start_epoch=1;successive=0;elapsed=0.
    checkpoint=LOCAL/'phase0'/f'{experiment}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    resume=checkpoint.with_suffix('.resume.pt')
    if resume.exists():
        state=torch.load(resume,map_location=DEVICE,weights_only=False)
        assert state['signature']==sig['signature']
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch']+1;history=state['history'];gradient_rows=state['gradients']
        successive=state['successive'];elapsed=state['elapsed'];rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if DEVICE=='cuda':torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    initial_activations=activation_audit(model,tensors)
    if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
    sync();began=time.perf_counter();last_logits=None
    for epoch in range(start_epoch,CONFIG['tiny_maximum_epochs']+1):
        model.train();order=rng.permutation(size);online=[]
        for group_start in range(0,size,RESOURCE['effective_batch_participants']):
            group=order[group_start:group_start+RESOURCE['effective_batch_participants']]
            optimizer.zero_grad(set_to_none=True)
            before={k:p.detach().clone() for k,p in model.named_parameters()} if epoch<=2 or epoch%10==0 else None
            for position in range(0,len(group),RESOURCE['micro_batch_participants']):
                indices=group[position:position+RESOURCE['micro_batch_participants']]
                logit=model(collate_people([tensors[i] for i in indices]))
                label=torch.tensor(y[chosen[indices]],device=DEVICE,dtype=torch.float32)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,label)
                assert torch.isfinite(loss), 'Nonfinite diagnostic objective'
                (loss*len(indices)/len(group)).backward();online.extend([float(loss.detach())]*len(indices))
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            if before is not None:gradient_rows.extend(gradient_summary(model,before,epoch,group_start))
        last_logits,_=predictions(model,tensors);p=1/(1+np.exp(-last_logits))
        bce=float(np.mean(np.logaddexp(0,last_logits)-y[chosen]*last_logits))
        row=dict(epoch=epoch,online_train_bce=float(np.mean(online)),eval_train_bce=bce,
            **metrics(y[chosen],p),logit_mean=float(last_logits.mean()),logit_sd=float(last_logits.std()),
            predicted_positive=int((p>.5).sum()),learning_rate=CONFIG['tiny_learning_rate'])
        history.append(row)
        criterion=CONFIG['tiny_success'];successive=successive+1 if row['f1_macro']>=criterion['macro_f1'] and bce<=criterion['eval_bce'] else 0
        if epoch%10==0 or successive>=criterion['consecutive_epochs'] or epoch==CONFIG['tiny_maximum_epochs']:
            sync();duration=elapsed+time.perf_counter()-began
            torch.save(dict(signature=sig['signature'],model=model.state_dict(),optimizer=optimizer.state_dict(),
                normalizer=normalizer,fit_ids=ids[chosen].tolist(),epoch=epoch,history=history,gradients=gradient_rows,
                successive=successive,elapsed=duration,numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if DEVICE=='cuda' else None),resume)
            pd.DataFrame(history).to_csv(destination/'learning_curve.csv',index=False)
            pd.DataFrame(gradient_rows).to_csv(destination/'gradient_updates.csv',index=False)
            resource_check()
            print('TINY',family,size,'epoch',epoch,'evalBCE',round(bce,5),'F1',round(row['f1_macro'],4),flush=True)
        if successive>=criterion['consecutive_epochs']:break
    sync();duration=elapsed+time.perf_counter()-began
    logits,embeddings=predictions(model,tensors,capture=True)
    sensitivity=[]
    for batch_size in (1,2,8):
        changed,_=predictions(model,tensors,batch_size)
        error=float(np.max(np.abs(logits-changed)))
        assert error<1e-4,'Batch dependence exceeds numerical tolerance'
        sensitivity.append(dict(batch_size=batch_size,maximum_absolute_logit_difference=error))
    model.train()
    with torch.no_grad():train_logits=np.concatenate([model(collate_people(tensors[i:i+2])).cpu().numpy() for i in range(0,size,2)])
    model.eval();np.testing.assert_allclose(train_logits,logits,atol=1e-4,rtol=0)
    pd.DataFrame(activation_audit(model,tensors)).to_csv(destination/'final_activations.csv',index=False)
    pd.DataFrame(initial_activations).to_csv(destination/'initial_activations.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(destination/'batch_sensitivity.csv',index=False)
    pd.DataFrame(dict(participant_id=ids[chosen],label=y[chosen],logit=logits,probability=1/(1+np.exp(-logits)))).to_csv(destination/'training_predictions.csv',index=False)
    np.savez_compressed(destination/'participant_embeddings.npz',ids=ids[chosen],z=embeddings)
    torch.save(dict(state=model.cpu().state_dict(),normalizer=normalizer,architecture=model.config,
        fit_ids=ids[chosen].tolist(),signature=sig['signature']),checkpoint)
    result=dict(experiment_id=experiment,signature=sig['signature'],complete=True,
        memorized=successive>=criterion['consecutive_epochs'],scope='Training-only engineering diagnostic; no generalization estimate',
        family=family,size=size,epochs=epoch,train_metrics=metrics(y[chosen],1/(1+np.exp(-logits))),eval_train_bce=history[-1]['eval_train_bce'],
        fit_ids=ids[chosen].tolist(),outer_test_evaluated=False,dropout=0.,weight_decay=0.,
        parameters=sum(p.numel() for p in model.parameters()),train_seconds=duration,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if DEVICE=='cuda' else 0,
        embedding_variance_mean=float(embeddings.var(0).mean()),mean_pair_distance=float(pdist(embeddings).mean()),
        logit_sd=float(logits.std()),checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint))
    write(target,result)
    ledger(experiment,training_status='completed',completion_status='completed',best_checkpoint=result['checkpoint'],
        validation_metrics=result['train_metrics'],parameter_count=result['parameters'],gpu_peak_memory=result['peak_gpu_bytes'],
        training_cost=duration,artifact_paths=[destination.relative_to(ROOT).as_posix()],
        scientific_interpretation='Tiny Train set memorized; optimization capacity only' if result['memorized'] else 'Failed to meet memorization criterion; investigate before expensive research fits')
    del tensors,model,optimizer
    if DEVICE=='cuda':torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--families',nargs='+',default=CONFIG['tiny_models']);parser.add_argument('--sizes',nargs='+',type=int,default=CONFIG['tiny_sizes']);args=parser.parse_args()
    with gpu_worker():
        for size in args.sizes:
            for family in args.families:fit_tiny(family,size)
    print('TINY MEMORIZATION EXPERIMENTS FINISHED',flush=True)


if __name__=='__main__':main()
