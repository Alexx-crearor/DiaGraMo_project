"""Explain fixed outer-fold models; never tune or refit when checkpoints exist."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('NUMBA_CACHE_DIR',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'tmp/numba_xai'))
import json
import time
import warnings
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from threadpoolctl import threadpool_limits
from xai_common import ROOT,OUT,FUSION,SCENARIOS,N_REPEATS,BACKGROUND,load_data,initialize,load_models,sha,write_json
from diagramo.explain import feature_mapping,align_shap,validate_additivity,aggregate_shap,fusion_mechanism,outcomes,linear_logit_contributions,permute_block
from evaluate import metrics

warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names')


def mapping_audit(X,ids,schema):
    from prepare import TASKS,canonical
    from diagramo.features import kinematic
    if TASKS!=schema['tasks']: raise ValueError('Schema task order differs from actual preparation code')
    rows=[]
    # Fixed source sentinels, independent of labels/predictions/attributions.
    for i in (0,len(ids)//2,len(ids)-1):
        p=ROOT/'data/clean_data'/f'{ids[i]}_merged.json'
        data=json.loads(p.read_text(encoding='utf-8')); tasks={}
        for key,obj in data.items():
            task=canonical(key)
            if task in TASKS:
                if task in tasks: raise ValueError('Duplicate source task')
                tasks[task]=obj['data']
        reconstructed=[]
        for task in TASKS:
            values,names=kinematic(tasks[task])
            if names!=schema['features_per_task']: raise ValueError('Source feature order differs from schema')
            reconstructed.append(values)
        diff=float(np.max(np.abs(np.stack(reconstructed).reshape(-1)-X[i])))
        if diff>1e-6: raise ValueError('Source task-feature flattening does not reproduce cached row')
        rows.append(dict(participant_id=ids[i],source=str(p.relative_to(ROOT)),source_sha256=sha(p),
                         tasks_checked=len(TASKS),features_checked=1036,max_abs_difference=diff))
    result=dict(task_order_source='data/derived/schema.json and scripts/prepare.py TASKS/reshape',
                feature_order_source='actual features.py kinematic() output checked against schema',
                sentinel_rule='first, middle, last participant; chosen without labels',sentinels=rows,
                n_tasks=14,n_features_per_task=74,total_features=1036,
                groups='12 actual source prefixes, not an assumed paper taxonomy')
    write_json(OUT/'mapping_audit.json',result)


def positive_explanation(explainer,X):
    e=explainer(X,check_additivity=True)
    if e.values.ndim==3:
        if e.values.shape[-1]!=2: raise ValueError('Unexpected output classes')
        e=e[...,1]
    if e.values.ndim!=2 or e.values.shape!=X.shape: raise ValueError('Unexpected TreeExplainer shape')
    return e


def run_fold(f,arrays,ids,y,folds,splits,mapping,sig):
    target=OUT/'folds'/f'fold_{f}.json'
    if target.exists():
        r=json.loads(target.read_text())
        if r['settings_sha256']!=sig: raise ValueError('Fold signature differs')
        for p,h in r['outputs_sha256'].items():
            if sha(OUT/p)!=h: raise ValueError('Fold result changed')
        print('REUSE XAI',f,flush=True); return
    start=time.perf_counter(); tr,te=splits[f]
    bundle,combiners,record,stage,reconstructed=load_models(f,arrays,ids,y,tr)
    if record['test_indices']!=te.tolist() or record['test_participants']!=ids[te].tolist(): raise ValueError('Outer Test differs')
    with np.load(FUSION/'stages'/f'outer_{f}.npz') as z: saved={k:z[k] for k in z.files}
    bank=bundle['banks']['K']; pipe=bundle['models']['K__none__LightGBM']; model=pipe[-1]
    A=bank.transform_many(arrays['K'][te])['none']; pred=pipe.predict(A); prob=pipe.predict_proba(A)[:,1]
    np.testing.assert_array_equal(pred,saved['K__none__LightGBM'][:,0])
    np.testing.assert_allclose(prob,saved['K__none__LightGBM'][:,1],atol=1e-12,rtol=1e-10)
    original=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv').iloc[te]
    if not np.array_equal(original.participant_id,ids[te]) or not np.array_equal(original.label,y[te]): raise ValueError('Baseline OOF identity')
    np.testing.assert_array_equal(original.prediction,pred)
    np.testing.assert_allclose(original.probability,prob,atol=1e-12,rtol=1e-10)
    bg_indices=np.sort(np.random.default_rng(42).choice(tr,size=BACKGROUND,replace=False))
    B=bank.transform_many(arrays['K'][bg_indices])['none']
    primary=positive_explanation(shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw'),A)
    raw=positive_explanation(shap.TreeExplainer(model,data=B,feature_perturbation='interventional',model_output='raw'),A)
    probability=positive_explanation(shap.TreeExplainer(model,data=B,feature_perturbation='interventional',model_output='probability'),A)
    margin=model.predict(A,raw_score=True)
    raw_error=validate_additivity(primary.values,primary.base_values,margin)
    interventional_error=validate_additivity(raw.values,raw.base_values,margin)
    probability_error=validate_additivity(probability.values,probability.base_values,prob)
    np.testing.assert_allclose(expit(margin),prob,atol=1e-12)
    # LightGBM binary raw score is verified as log-odds, not assumed for arbitrary models.
    np.testing.assert_allclose(margin,np.log(prob)-np.log1p(-prob),atol=1e-10)
    keep=bank.pre_[1].get_support(indices=True)
    attrs=OUT/'attributions'; attrs.mkdir(exist_ok=True)
    p=attrs/f'kinematic_fold_{f}.npz'
    np.savez_compressed(p,indices=te,participant_ids=ids[te],raw_path=align_shap(primary.values,keep),
        raw_interventional=align_shap(raw.values,keep),probability=align_shap(probability.values,keep),
        base_raw_path=primary.base_values,base_raw_interventional=raw.base_values,base_probability=probability.base_values,
        raw_score=margin,prediction=pred,predicted_probability=prob,feature_values=arrays['K'][te],kept_indices=keep)
    outputs=[p]
    fusion_rows=[]; cwt_summary=[]
    settings=json.loads((FUSION/'settings.json').read_text()); evaluated=pd.read_csv(FUSION/'nested_oof.csv')
    Ctest=bundle['banks']['C'].transform_many(arrays['C'][te])
    Cbackground=bundle['banks']['C'].transform_many(arrays['C'][bg_indices])
    for scenario in SCENARIOS:
        evaluated_result=record['results'][scenario]; config=settings['grid'][evaluated_result['candidate']]
        if config['method']=='baseline': raise ValueError('This fixed study expects evaluated KC candidates with a CWT branch')
        options=config['options']
        if options[0]!='K__none__LightGBM' or len(options)!=2: raise ValueError('Unexpected fusion branch order')
        ckey=options[1]; branch,spec,clf=ckey.split('__')
        if branch!='C' or clf!='LogisticRegression': raise ValueError('Unexpected CWT model')
        cp=bundle['models'][ckey]; Z=Ctest[spec]; pc=cp.predict_proba(Z)[:,1]
        np.testing.assert_allclose(pc,saved[ckey][:,1],atol=1e-12,rtol=1e-10)
        combiner=combiners['models'][scenario]
        fused=combiner.predict_proba(np.column_stack([prob,pc]))[:,1]
        mechanism=fusion_mechanism(prob,pc,combiner.metadata(),config['method'])
        np.testing.assert_allclose(mechanism['probability'],fused,atol=1e-12)
        np.testing.assert_allclose(fused,evaluated_result['probability'],atol=1e-12)
        frame=evaluated[(evaluated.scenario==scenario)&(evaluated.fold==f)].set_index('participant_id').loc[ids[te]]
        np.testing.assert_allclose(frame.probability,fused,atol=1e-12)
        final_pred=(fused>=.5).astype(int)
        np.testing.assert_array_equal(frame.prediction,final_pred)
        status=outcomes(y[te],pred,final_pred)
        for j,i in enumerate(te):
            fusion_rows.append(dict(scenario=scenario,fold=f,participant_id=ids[i],label=int(y[i]),candidate=evaluated_result['candidate'],
                cwt_reducer=spec,cwt_dimensions=Z.shape[1],kinematic_probability=float(prob[j]),cwt_probability=float(pc[j]),
                fusion_probability=float(fused[j]),kinematic_prediction=int(pred[j]),cwt_prediction=int(pc[j]>=.5),
                fusion_prediction=int(final_pred[j]),effective_weight_kinematic=float(mechanism['effective_weight_kinematic'][j]),
                effective_weight_cwt=float(mechanism['effective_weight_cwt'][j]),
                nominal_weights=json.dumps(combiner.weights_.tolist()),gate_applicable=config['method']=='adaptive',
                gate_active=bool(mechanism['gate_active'][j]) if config['method']=='adaptive' else None,
                gate_threshold=combiner.gate if config['method']=='adaptive' else None,
                budget=combiner.budget if config['method']=='adaptive' else None,
                delta_cwt=float(mechanism['delta_cwt'][j]),kinematic_uncertainty_distance=float(abs(prob[j]-.5)),
                moves_toward_observed_label=float((2*y[i]-1)*mechanism['delta_cwt'][j]),outcome=status[j]))
        phi,base=linear_logit_contributions(cp[-1],Z,Cbackground[spec])
        q=attrs/f'cwt_{scenario}_fold_{f}.npz'
        np.savez_compressed(q,indices=te,values=phi,base_value=base,decision_function=cp[-1].decision_function(Z),
                            predicted_probability=pc,reference_mean=Cbackground[spec].mean(axis=0))
        outputs.append(q)
        # PCA axes differ across folds. Never pool PC1 from different models.
        ranking=np.argsort(-np.mean(np.abs(phi),axis=0),kind='stable')[:20]
        for rank,k in enumerate(ranking,1):
            cwt_summary.append(dict(scenario=scenario,fold=f,reducer=spec,component_index=int(k),rank=rank,
                component=(f'fold{f}_PC{k+1:03d}' if spec.startswith('pca') else f'preprocessed_embedding_coordinate_{k:05d}'),
                mean_abs_logit_contribution=float(np.abs(phi[:,k]).mean()),mean_logit_contribution=float(phi[:,k].mean()),
                biological_interpretation_available=False))
    fusion_path=OUT/'folds'/f'fusion_{f}.csv'; fusion_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(fusion_rows).to_csv(fusion_path,index=False); outputs.append(fusion_path)
    cwt_path=OUT/'folds'/f'cwt_components_{f}.csv'
    pd.DataFrame(cwt_summary).to_csv(cwt_path,index=False); outputs.append(cwt_path)
    # Joint permutation of whole task/group blocks; no new model fit or threshold.
    blocks=[]
    for level in ('task','group'):
        for name,frame in mapping.groupby(level,sort=False): blocks.append((level,name,frame.feature_index.to_numpy()))
    rng=np.random.default_rng(42+1000*f); permutation_indices=np.empty((len(blocks),N_REPEATS,len(te)),dtype=np.int16)
    perm_prob=np.empty_like(permutation_indices,dtype=float); perm_pred=np.empty_like(permutation_indices,dtype=np.int8)
    baseline=metrics(y[te],pred,prob); permutation_rows=[]
    for b,(level,name,columns) in enumerate(blocks):
        for rep in range(N_REPEATS):
            order=rng.permutation(len(te)); permutation_indices[b,rep]=order
            altered=permute_block(arrays['K'][te],columns,order)
            T=bank.transform_many(altered)['none']
            pp=pipe.predict_proba(T)[:,1]; prediction=pipe.predict(T)
            perm_prob[b,rep]=pp; perm_pred[b,rep]=prediction
            score=metrics(y[te],prediction,pp)
            permutation_rows.append(dict(fold=f,level=level,block=name,repeat=rep,n_features=len(columns),
                **{'permuted_'+m:score[m] for m in ('accuracy','f1_macro','auc','balanced_accuracy')},
                **{'drop_'+m+'_pp':100*(baseline[m]-score[m]) for m in ('accuracy','f1_macro','auc','balanced_accuracy')}))
    perm_dir=OUT/'permutation'; perm_dir.mkdir(exist_ok=True)
    q=perm_dir/f'fold_{f}.npz'; np.savez_compressed(q,permutation_indices=permutation_indices,
                                                 probability=perm_prob,prediction=perm_pred,indices=te)
    outputs.append(q)
    q=perm_dir/f'fold_{f}.csv'; pd.DataFrame(permutation_rows).to_csv(q,index=False); outputs.append(q)
    write_json(target,dict(fold=f,settings_sha256=sig,train_participants=ids[tr].tolist(),test_participants=ids[te].tolist(),
        test_indices=te.tolist(),background_participants=ids[bg_indices].tolist(),background_indices=bg_indices.tolist(),
        model_checkpoint=str((FUSION/stage['checkpoint']).relative_to(ROOT)),model_checkpoint_sha256=stage['checkpoint_sha256'],
        reconstructed=reconstructed,kinematic_fit_participants=bank.fit_ids_,removed_constant_features=1036-len(keep),
        shap_errors=dict(path_raw=raw_error,interventional_raw=interventional_error,interventional_probability=probability_error),
        shap_units=dict(path_raw='log-odds for group=1',interventional_raw='log-odds for group=1',interventional_probability='probability group=1'),
        permutation_blocks=[dict(level=l,name=n,feature_indices=ix.tolist()) for l,n,ix in blocks],
        permutation_repeats=N_REPEATS,seconds=time.perf_counter()-start,
        outputs_sha256={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in outputs}))
    print('XAI OUTER COMPLETE',f,round(time.perf_counter()-start,1),'s',flush=True)


def main():
    sig=initialize(); arrays,ids,y,folds,splits=load_data()
    schema=json.loads((ROOT/'data/derived/schema.json').read_text()); mapping=feature_mapping(schema)
    mapping.to_csv(OUT/'feature_mapping.csv',index=False); mapping_audit(arrays['K'],ids,schema)
    folds.to_csv(OUT/'fold_assignments.csv',index=False)
    with threadpool_limits(limits=2):
        for f in range(6): run_fold(f,arrays,ids,y,folds,splits,mapping,sig)
    print('All six XAI folds completed; run report and final validation next.',flush=True)


if __name__=='__main__': main()
