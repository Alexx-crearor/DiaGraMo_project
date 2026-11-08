"""Nested fusion ablation with a separate cross-fit inside every inner Train.

Only inner predictions enter choose_scenarios. Outer labels are used solely for
reported metrics. Checkpoints and immutable signatures allow safe continuation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import gc
import json
import time
import warnings
import concurrent.futures
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from fusion_common import ROOT,OUT,GRID,VARIANTS,BRANCH_OPTIONS,METRICS,option,load_data,signature,sha,write_json
from evaluate import build_pipeline,classifiers,metrics
from diagramo.fusion import BranchBank,ProbabilityFusion,choose_inner,check_crossfit

warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names')


def cv(ids,y):
    splits=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=42).split(ids,y))
    check_crossfit(ids,splits)
    return splits


def stage(arrays,ids,y,tr,te,tag,sig,early_names=None,save_models=False):
    """Fit a bank on tr ONLY; te labels are never accessed, including for cache."""
    path=OUT/'stages'/tag
    audit_path=path.with_suffix('.json'); pred_path=path.with_suffix('.npz')
    if audit_path.exists() and pred_path.exists():
        record=json.loads(audit_path.read_text())
        if record['settings_sha256']!=sig or record['pred_sha256']!=sha(pred_path): raise ValueError('Stage cache changed')
        if record['train_participants']!=ids[tr].tolist() or record['test_participants']!=ids[te].tolist():
            raise ValueError('Stage split mismatch')
        with np.load(pred_path) as z:
            return {k:z[k] for k in z.files},record
    if set(ids[tr])&set(ids[te]): raise ValueError('Leaking split')
    began=time.perf_counter(); banks={}; projected={}; fitted={}; costs=[]; predictions={}; dimensions={}
    models=classifiers(42)
    for b,opts in BRANCH_OPTIONS.items():
        start=time.perf_counter()
        specs=list(dict.fromkeys(s for s,c in opts))
        bank=BranchBank().fit(arrays[b][tr],y[tr],ids[tr],specs)
        train=bank.transform_many(arrays[b][tr]); test=bank.transform_many(arrays[b][te])
        banks[b]=bank; projected[b]=(train,test)
        costs.append(dict(component='bank_'+b,seconds=time.perf_counter()-start))
        for spec,clf in opts:
            key=option(b,spec,clf); start=time.perf_counter()
            pipe=build_pipeline(clf,models[clf],preprocess=False).fit(train[spec],y[tr])
            fit_seconds=time.perf_counter()-start; start=time.perf_counter()
            predictions[key]=np.column_stack([pipe.predict(test[spec]),pipe.predict_proba(test[spec])[:,1]])
            dimensions[key]=train[spec].shape[1]
            costs.append(dict(component=key,seconds=fit_seconds,predict_seconds=time.perf_counter()-start))
            if save_models: fitted[key]=pipe
    for name in early_names or []:
        c=GRID[name]; start=time.perf_counter()
        if c['method']=='original':
            A=np.concatenate([arrays[b][tr] for b in c['branches']],axis=1)
            B=np.concatenate([arrays[b][te] for b in c['branches']],axis=1)
            pipe=build_pipeline(c['classifier'],models[c['classifier']])
        else:
            A=np.concatenate([projected[b][0][s] for b,s in zip(c['branches'],c['specs'])],axis=1)
            B=np.concatenate([projected[b][1][s] for b,s in zip(c['branches'],c['specs'])],axis=1)
            pipe=build_pipeline(c['classifier'],models[c['classifier']],preprocess=False)
        pipe.fit(A,y[tr]); fit_seconds=time.perf_counter()-start; start=time.perf_counter()
        predictions[name]=np.column_stack([pipe.predict(B),pipe.predict_proba(B)[:,1]])
        dimensions[name]=A.shape[1]
        costs.append(dict(component=name,seconds=fit_seconds,predict_seconds=time.perf_counter()-start))
        if save_models: fitted[name]=pipe
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(pred_path,**predictions)
    record=dict(settings_sha256=sig,train_participants=ids[tr].tolist(),test_participants=ids[te].tolist(),
                banks={b:bank.audit() for b,bank in banks.items()},dimensions=dimensions,costs=costs,
                seconds=time.perf_counter()-began,pred_sha256=sha(pred_path),early_names=list(early_names or []))
    if save_models:
        modelpath=OUT/'checkpoints'/f'{tag}.joblib'; modelpath.parent.mkdir(exist_ok=True)
        joblib.dump(dict(banks=banks,models=fitted,grid={n:GRID[n] for n in early_names or []},
                         train_participants=ids[tr].tolist(),settings_sha256=sig),modelpath,compress=3)
        record['checkpoint']=str(modelpath.relative_to(OUT)); record['checkpoint_sha256']=sha(modelpath)
    write_json(audit_path,record)
    print('STAGE',tag,round(record['seconds'],1),'s',flush=True)
    return predictions,record


def crossfit(arrays,ids,y,tag,sig):
    """Only the passed Train subset and its labels are available here."""
    result={option(b,s,c):np.empty((len(y),2)) for b,opts in BRANCH_OPTIONS.items() for s,c in opts}
    tags=[]
    for k,(tr,va) in enumerate(cv(ids,y)):
        name=f'{tag}_crossfit_{k}'
        preds,audit=stage(arrays,ids,y,tr,va,name,sig)
        for key in result: result[key][va]=preds[key]
        tags.append(name)
    return result,tags


def predict_candidate(c,validation,train_oof,train_y):
    if c['method'] in ('baseline','single'):
        return validation[c['options'][0]],None
    P=np.column_stack([train_oof[k][:,1] for k in c['options']])
    Q=np.column_stack([validation[k][:,1] for k in c['options']])
    model=ProbabilityFusion(**c['fusion']).fit(P,train_y)
    probability=model.predict_proba(Q)[:,1]
    return np.column_stack([probability>=.5,probability]),model


def aggregate_inner(scores):
    df=pd.DataFrame(scores)
    rows=[]
    for name,g in df.groupby('candidate',sort=False):
        if len(g)!=3 or set(g.inner_fold)!={0,1,2}: raise ValueError('Incomplete inner grid')
        rows.append(dict(candidate=name,mean_features=float(g.n_features.mean()),
                         **{'mean_'+m:float(g[m].mean()) for m in METRICS}))
    return rows


def choose_scenarios(ranking):
    byname={r['candidate']:r for r in ranking}; chosen={'Kinematic_LightGBM':byname['K_only']}
    for method in ('original','compact','late','adaptive'):
        for variant in VARIANTS:
            pool=[r for r in ranking if GRID[r['candidate']]['method']==method and GRID[r['candidate']]['variant']==variant]
            if not pool: continue
            chosen[f'{method}__{variant}']=choose_inner(pool,byname['K_only'] if method=='adaptive' else None)
            if method=='late':
                chosen[f'equal__{variant}']=choose_inner([r for r in pool if GRID[r['candidate']]['fusion']['kind']=='equal'])
    for b in ('M','G','A','C'):
        chosen['single__'+b]=choose_inner([r for r in ranking if GRID[r['candidate']]['method']=='single' and GRID[r['candidate']]['variant']==b])
    pool=[r for r in ranking if GRID[r['candidate']]['method'] in ('baseline','original','compact','late','adaptive')
          and not GRID[r['candidate']]['variant'].endswith('adapter')]
    chosen['all_selected']=choose_inner(pool,byname['K_only'])
    return chosen


def train_selection(arrays,ids,y,fold,sig):
    """Select ONLY using outer Train. It is impossible to access outer Test here."""
    target=OUT/'selections'/f'fold_{fold}.json'
    if target.exists():
        result=json.loads(target.read_text())
        if result['settings_sha256']!=sig: raise ValueError('Selection signature changed')
        return result
    scores=[]; inner_predictions={name:np.empty((len(y),2)) for name in GRID}
    base_oof={option(b,s,c):np.empty((len(y),2)) for b,opts in BRANCH_OPTIONS.items() for s,c in opts}
    audits=[]
    early=[n for n,c in GRID.items() if c['method'] in ('original','compact')]
    for k,(tr,va) in enumerate(cv(ids,y)):
        tag=f'outer_{fold}_inner_{k}'
        validation,audit=stage(arrays,ids,y,tr,va,tag,sig,early_names=early)
        subset={b:X[tr] for b,X in arrays.items()}
        oof,tags=crossfit(subset,ids[tr],y[tr],tag,sig)
        for key in base_oof: base_oof[key][va]=validation[key]
        for name,c in GRID.items():
            if c['method'] in ('original','compact'):
                pred=validation[name]; d=audit['dimensions'][name]
            else:
                pred,model=predict_candidate(c,validation,oof,y[tr])
                d=sum(audit['dimensions'][o] for o in c['options'])
            inner_predictions[name][va]=pred
            scores.append(dict(candidate=name,inner_fold=k,n_features=d,
                               **metrics(y[va],pred[:,0].astype(int),pred[:,1])))
        audits.append(dict(inner_fold=k,train_participants=ids[tr].tolist(),validation_participants=ids[va].tolist(),
                           stage=tag,crossfit_stages=tags,combiner_fit_participants=ids[tr].tolist()))
        print('INNER COMPLETE',fold,k,flush=True)
    ranking=aggregate_inner(scores); selections=choose_scenarios(ranking)
    dest=OUT/'selections'; dest.mkdir(exist_ok=True)
    predpath=dest/f'fold_{fold}_inner_predictions.npz'
    np.savez_compressed(predpath,**inner_predictions)
    basepath=dest/f'fold_{fold}_base_crossfit.npz'
    np.savez_compressed(basepath,**base_oof)
    result=dict(fold=fold,settings_sha256=sig,train_participants=ids.tolist(),
                inner_audits=audits,inner_scores=scores,ranking=ranking,selections=selections,
                inner_predictions_sha256=sha(predpath),base_crossfit_sha256=sha(basepath))
    write_json(target,result)
    return result


def outer_job(f,arrays,ids,y,splits,sig):
    target=OUT/'folds'/f'fold_{f}.json'
    if target.exists():
        if json.loads(target.read_text())['settings_sha256']!=sig: raise ValueError('Outer signature changed')
        print('REUSE OUTER',f,flush=True); return
    began=time.perf_counter(); tr,te=splits[f]
    selected=train_selection({b:X[tr] for b,X in arrays.items()},ids[tr],y[tr],f,sig)
    early=list(dict.fromkeys(r['candidate'] for r in selected['selections'].values()
                            if GRID[r['candidate']]['method'] in ('original','compact')))
    validation,audit=stage(arrays,ids,y,tr,te,f'outer_{f}',sig,early_names=early,save_models=True)
    with np.load(OUT/'selections'/f'fold_{f}_base_crossfit.npz') as z: oof={k:z[k] for k in z.files}
    results={}; meta_models={}
    for scenario,row in selected['selections'].items():
        name=row['candidate']; c=GRID[name]
        if c['method'] in ('original','compact'):
            pred=validation[name]; model=None; d=audit['dimensions'][name]
        else:
            pred,model=predict_candidate(c,validation,oof,y[tr])
            d=sum(audit['dimensions'][o] for o in c['options'])
        result=dict(candidate=name,n_features=d,prediction=pred[:,0].astype(int).tolist(),probability=pred[:,1].tolist(),
                    metrics=metrics(y[te],pred[:,0].astype(int),pred[:,1]))
        if model is not None:
            result['combiner']=model.metadata(); meta_models[scenario]=model
            result['combiner_fit_participants']=ids[tr].tolist()
            if c['method']=='adaptive':
                pk=validation[c['options'][0]][:,1]
                result['gate_active']=(np.abs(pk-.5)<=c['fusion']['gate']).tolist()
        results[scenario]=result
    metapath=OUT/'checkpoints'/f'outer_{f}_combiners.joblib'
    joblib.dump(dict(models=meta_models,selections=selected['selections'],settings_sha256=sig,
                     train_participants=ids[tr].tolist()),metapath,compress=3)
    write_json(target,dict(fold=f,settings_sha256=sig,train_participants=ids[tr].tolist(),test_participants=ids[te].tolist(),
                test_indices=te.tolist(),results=results,seconds=time.perf_counter()-began,
                combiner_checkpoint=str(metapath.relative_to(OUT)),combiner_checkpoint_sha256=sha(metapath)))
    print('OUTER COMPLETE',f,round(time.perf_counter()-began,1),'s',flush=True)
    gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,choices=[1,2],default=2)
    args=p.parse_args()
    sig=signature(); arrays,ids,y,folds,splits=load_data()
    folds.to_csv(OUT/'fold_assignments.csv',index=False)
    with threadpool_limits(limits=2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(outer_job,f,arrays,ids,y,splits,sig) for f in range(6)]
            for future in concurrent.futures.as_completed(futures): future.result()
