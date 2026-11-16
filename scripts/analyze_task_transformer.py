"""Held-out robustness, logit IG and independent task/group perturbation checks."""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from task_transformer_common import ROOT, OUT, CONFIG, FAMILIES, METRICS, data, cache, device, score, sha, write, features_for
from train_task_transformer import load_model, predict
from diagramo.task_transformer import integrated_gradients
from diagramo.kinematic_xai import feature_mapping, blocks, explain_pipeline


def segmented_ig(model,Z,present,dev,order=64):
    """Resolve narrow near-reference transitions without changing the fitted model.

    Extremely large held-out standardized coordinates can put all logit change
    before the first global Gauss node. Geometric alpha panels resolve that region;
    this is numerical integration, not model/feature/threshold selection.
    """
    nodes,weights=np.polynomial.legendre.leggauss(order)
    phi=np.zeros_like(Z,dtype=np.float64);all_panels=[]
    for i,example in enumerate(Z):
        magnitude=max(1.,float(np.max(np.abs(example))))
        start=min(1e-3,1e-3/magnitude)
        n=max(2,int(np.ceil(-np.log10(start)))+1)
        edges=np.r_[0.,np.geomspace(start,1.,n)]
        all_panels.append(len(edges)-1)
        batch=torch.tensor(example[None],device=dev)
        mask=torch.tensor(np.asarray(present,bool)[None],device=dev)
        total=torch.zeros_like(batch,dtype=torch.float64)
        for left,right in zip(edges[:-1],edges[1:]):
            alpha=left+(nodes+1)*(right-left)/2;w=weights*(right-left)/2
            for j in range(0,order,8):
                point=(torch.tensor(alpha[j:j+8],device=dev,dtype=batch.dtype)[:,None,None]*batch).detach().requires_grad_(True)
                gradient=torch.autograd.grad(model(point,mask.expand(len(point),-1)).sum(),point)[0]
                total+=(gradient.double()*torch.tensor(w[j:j+8],device=dev)[:,None,None]).sum(0,keepdim=True)
        phi[i]=(batch.double()*total).detach().cpu().numpy()[0]
    phi*=np.asarray(present)[None,:,None]
    with torch.no_grad():
        x=torch.tensor(Z,device=dev);mask=torch.tensor(np.asarray(present,bool),device=dev).expand(len(Z),-1)
        logits=model(x,mask).cpu().numpy();base=model(torch.zeros_like(x),mask).cpu().numpy()
    return phi,base,logits,phi.sum((1,2))-(logits-base),all_panels


def stable_ig(model, Z, present, dev):
    absolute=CONFIG['integrated_gradients_absolute_tolerance']
    relative=CONFIG['integrated_gradients_relative_tolerance']
    for steps in CONFIG['integrated_gradients_steps']:
        phi,base,logits,error=integrated_gradients(model,Z,present,steps=steps,device=dev)
        tolerance=absolute+relative*np.abs(logits-base)
        if np.all(np.abs(error)<=tolerance):
            return dict(phi=phi.reshape(len(Z),1036),base=base,logits=logits,error=error,steps=steps,
                        absolute_tolerance=absolute,relative_tolerance=relative)
    failed=np.abs(error)>tolerance
    original_error=error.copy()
    # Retain converged global quadrature results; refine only unresolved participants.
    for order in (32,64,128,256):
        refined,ref,log,err,panels=segmented_ig(model,Z[failed],present,dev,order)
        phi[failed]=refined;base[failed]=ref;logits[failed]=log;error[failed]=err
        if np.all(np.abs(error)<=absolute+relative*np.abs(logits-base)):
            return dict(phi=phi.reshape(len(Z),1036),base=base,logits=logits,error=error,steps=steps,
                absolute_tolerance=absolute,relative_tolerance=relative,
                refinement='geometric alpha panels for unresolved participants; no change to model or tolerance',
                refined_indices=np.flatnonzero(failed),global_quadrature_error=original_error,
                panel_order=order,panel_counts=np.asarray(panels))
    raise ValueError(f'IG segmented quadrature did not converge: max residual {np.max(np.abs(error))}')


def run_fold(f, dev):
    destination=OUT/'analysis'/f'fold_{f}'
    if (destination/'complete.json').exists():
        print('REUSE ANALYSIS',f,flush=True);return
    X,y,ids,schema,assignments,original,splits=data()
    tr,te=splits[f]
    bank=cache()
    mapping=feature_mapping(schema['tasks'],schema['features_per_task'])
    destination.mkdir(parents=True,exist_ok=True)
    rows_robust,rows_permutation,rows_mask,rows_attention,rows_shift,rows_cost,rows_zero=[],[],[],[],[],[],[]
    baselines={}
    entries={}
    for label in ['lightgbm','lightgbm_without_text',*FAMILIES]:
        info=json.loads((OUT/'folds'/f'fold_{f}'/'outer'/f'{label}_seed_42.json').read_text())
        entries[label]=info
        assert sha(ROOT/info['checkpoint'])==info['checkpoint_sha256']
        if label.startswith('lightgbm'):
            model=joblib.load(ROOT/info['checkpoint'])
            columns=np.asarray(info['columns'])
            feature=X
            present=np.zeros(14,bool);present[np.unique(columns//74)]=True
            def forward(A,mask=None):
                test=np.array(A,copy=True)
                if mask is not None:
                    missing=np.flatnonzero(np.repeat(~mask,74))
                    # Map active baseline columns to their original raw positions.
                    local=np.flatnonzero(np.isin(columns,missing))
                    test[:,columns[local]]=model[0].statistics_[local]
                return model.predict_proba(test[:,columns])[:,1]
        else:
            model,bundle=load_model(ROOT/info['checkpoint'],dev)
            pre=bundle['preprocessor'];present=np.asarray(bundle['present'],dtype=bool)
            feature=features_for(label,bank)
            def forward(A,mask=None):
                return predict(model,pre,A,present if mask is None else mask,dev)[0]
        clean=forward(feature[te])
        timings=[]
        for repeat in range(20):
            began=time.perf_counter(); measured=forward(feature[te]);timings.append(time.perf_counter()-began)
            np.testing.assert_allclose(measured,clean,atol=2e-6,rtol=0)
        cost=dict(fold=f,model=label,n_test=len(te),median_inference_seconds=float(np.median(timings)),
                  min_inference_seconds=float(np.min(timings)),repeats=20,
                  scope='cached Kinematic features -> preprocessing -> probability; excludes raw feature extraction')
        if label=='lightgbm':
            from task_transformer_common import factory
            began=time.perf_counter();timing_only=factory().fit(X[tr],y[tr]);cost['timing_only_train_seconds']=time.perf_counter()-began
            np.testing.assert_allclose(timing_only.predict_proba(X[te])[:,1],clean,atol=1e-12,rtol=0)
        rows_cost.append(cost)
        baselines[label]=score(y[te],clean)
        for nuisance in CONFIG['nuisances']:
            perturbed=bank['stress_'+nuisance] if label.startswith('lightgbm') else features_for(label,bank,nuisance)
            prob=forward(perturbed[te])
            s=score(y[te],prob)
            for i,index in enumerate(te):
                rows_robust.append(dict(fold=f,model=label,condition=nuisance,participant_id=ids[index],
                    label=int(y[index]),probability=prob[i],prediction=int(prob[i]>.5),clean_probability=clean[i],
                    probability_shift=prob[i]-clean[i]))
            rows_shift.append(dict(fold=f,model=label,condition=nuisance,**s,
                **{'delta_'+k:s[k]-baselines[label][k] for k in METRICS},mean_absolute_probability_shift=float(np.mean(np.abs(prob-clean)))))
        # Single-task deletion uses missing-token masks for nets, Train medians for trees.
        mask_conditions=[]
        for task_index,task in enumerate(schema['tasks']):
            mask_conditions.append(('one_task',task,0,np.array([task_index])))
        mask_conditions.append(('without_text_diagnostic','copy+dictation',0,
              np.array([schema['tasks'].index('copy'),schema['tasks'].index('dictation')])))
        for k in CONFIG['multiple_mask_counts']:
            for repeat in range(CONFIG['multiple_mask_repeats']):
                seed=int(np.random.SeedSequence([42,f,733,k,repeat]).generate_state(1)[0])
                removed=np.sort(np.random.default_rng(seed).choice(14,k,replace=False))
                mask_conditions.append((f'missing_{k}', '|'.join(schema['tasks'][i] for i in removed),repeat,removed))
        for kind,name,repeat,removed in mask_conditions:
            mask=present.copy();mask[removed]=False
            if not mask.any():
                raise ValueError('Predefined missing-task condition removed every selected task')
            prob=forward(feature[te],mask)
            for i,index in enumerate(te):
                rows_mask.append(dict(fold=f,model=label,kind=kind,removed=name,repeat=repeat,
                    present_count=int(mask.sum()),participant_id=ids[index],label=int(y[index]),
                    probability=prob[i],prediction=int(prob[i]>.5),clean_probability=clean[i]))
        for ti,task in enumerate(schema['tasks']):
            if label.startswith('lightgbm'):
                test=np.array(feature[te],copy=True)
                test[:,ti*74:(ti+1)*74]=X[tr,ti*74:(ti+1)*74].mean(axis=0,dtype=np.float64)
                prob=forward(test)
            else:
                Z=pre.transform(feature[te]);Z[:,ti]=0.
                with torch.no_grad():
                    z=torch.tensor(Z,device=dev)
                    m=torch.tensor(present,device=dev).expand(len(te),-1)
                    prob=torch.sigmoid(model(z,m)).cpu().numpy().astype(float)
            for i,index in enumerate(te):
                rows_zero.append(dict(fold=f,model=label,task=task,participant_id=ids[index],label=int(y[index]),
                    probability=prob[i],prediction=int(prob[i]>.5),clean_probability=clean[i]))
        # Paired shared permutations: same block, seed and participant ordering for all models.
        orders,probabilities,predictions=[],[],[]
        for li,level in enumerate(('task','group')):
            for bi,(block,columns_block) in enumerate(blocks(mapping,level).items()):
                for repeat in range(CONFIG['permutation_repeats']):
                    seed=int(np.random.SeedSequence([42,f,991,li,bi,repeat]).generate_state(1)[0])
                    order=np.random.default_rng(seed).permutation(len(te))
                    shuffled=np.array(feature[te],copy=True)
                    shuffled[:,columns_block]=feature[te][order][:,columns_block]
                    prob=forward(shuffled)
                    s=score(y[te],prob)
                    rows_permutation.append(dict(fold=f,model=label,level=level,block=block,repeat=repeat,seed=seed,
                         **s,**{'drop_'+k:baselines[label][k]-s[k] for k in METRICS}))
                    orders.append(order);probabilities.append(prob);predictions.append((prob>.5).astype(int))
        np.savez_compressed(destination/(label+'_permutations.npz'),orders=orders,probability=probabilities,prediction=predictions)
        if label.startswith('lightgbm'):
            if label=='lightgbm':
                exp=explain_pipeline(model,X[te])
                np.savez_compressed(destination/'lightgbm_shap.npz',phi=exp['phi'],base=exp['base'],logits=exp['raw_score'])
                clean_mass=np.abs(exp['phi']).reshape(-1,14,74).sum(2).mean(0)
                for nuisance in CONFIG['nuisances']:
                    perturbed=explain_pipeline(model,bank['stress_'+nuisance][te])
                    changed=np.abs(perturbed['phi']).reshape(-1,14,74).sum(2).mean(0)
                    rows_shift.append(dict(fold=f,model=label,condition=nuisance,attribution_rank_spearman=float(spearmanr(clean_mass,changed).statistic)))
        else:
            Z=pre.transform(feature[te])
            ig=stable_ig(model,Z,present,dev)
            np.savez_compressed(destination/(label+'_ig.npz'),ids=ids[te],present=present,**ig)
            prob,attention=predict(model,pre,feature[te],present,dev,return_attention=True)
            if attention['attention'].size:
                a=attention['attention'].mean(axis=(0,2))
                incoming=(a*present[None,:,None]).sum(axis=1)/present.sum()
                for i,index in enumerate(te):
                    for task_index,task in enumerate(schema['tasks']):
                        rows_attention.append(dict(fold=f,model=label,participant_id=ids[index],task=task,
                            incoming_attention=incoming[i,task_index],pool_weight=attention['pool_weights'][i,task_index]))
            if label in ('transformer','motor_normalized'):
                clean_mass=np.abs(ig['phi']).reshape(-1,14,74).sum(2).mean(0)
                for nuisance in CONFIG['nuisances']:
                    stress=pre.transform(features_for(label,bank,nuisance)[te])
                    stressed_ig=stable_ig(model,stress,present,dev)
                    np.savez_compressed(destination/(label+'_ig_'+nuisance+'.npz'),ids=ids[te],present=present,**stressed_ig)
                    changed=np.abs(stressed_ig['phi']).reshape(-1,14,74).sum(2).mean(0)
                    rows_shift.append(dict(fold=f,model=label,condition=nuisance,attribution_rank_spearman=float(spearmanr(clean_mass,changed).statistic)))
        print('ANALYSIS',f,label,'DONE',flush=True)
    pd.DataFrame(rows_robust).to_csv(destination/'robustness_oof.csv',index=False)
    pd.DataFrame(rows_shift).to_csv(destination/'robustness_folds.csv',index=False)
    pd.DataFrame(rows_permutation).to_csv(destination/'permutation.csv',index=False)
    pd.DataFrame(rows_mask).to_csv(destination/'masking_oof.csv',index=False)
    pd.DataFrame(rows_attention).to_csv(destination/'attention.csv',index=False)
    pd.DataFrame(rows_cost).to_csv(destination/'inference_cost.csv',index=False)
    pd.DataFrame(rows_zero).to_csv(destination/'feature_ablation_oof.csv',index=False)
    write(destination/'complete.json',dict(complete=True,fold=f,models=8,
         permutation_evaluations=len(rows_permutation),ig_models=6,stress_ig_models=12))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--folds',nargs='+',type=int,default=list(range(6)))
    args=ap.parse_args()
    with threadpool_limits(limits=4):
        for f in args.folds:run_fold(f,device())


if __name__=='__main__':main()
