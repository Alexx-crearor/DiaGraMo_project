"""Subject-disjoint outer evaluation and inner selection of frozen adaptations.

Original-bank results are reused exactly from the validated prior experiment.
Every new inner/outer split gets an independent ReductionBank. No fusion is run.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import concurrent.futures
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from embedding_common import ROOT,OUT,BRANCHES,MODES,CLASSIFIERS,REDUCERS,PRIMARY,initialize,sha,write_json
from evaluate import build_pipeline,classifiers,metrics
from diagramo.reduction import ReductionBank

warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names')


def load_subjects():
    with np.load(ROOT/'data/derived/kinematic.npz') as data:
        ids=data['ids'].astype(str); y=data['y']
    folds=pd.read_csv(ROOT/'results/cv/fold_assignments.csv')
    if len(ids)!=258 or len(set(ids))!=258: raise ValueError('258 unique participants required')
    if not np.array_equal(folds.participant_id,ids) or not np.array_equal(folds.label,y):
        raise ValueError('Original subject/label identity changed')
    splits=list(StratifiedKFold(n_splits=6,shuffle=True,random_state=42).split(ids,y))
    for f,(tr,te) in enumerate(splits):
        if set(ids[tr])&set(ids[te]) or not (folds.fold.iloc[te]==f).all():
            raise ValueError('Fold assignments differ from baseline')
    return ids,y,folds,splits


def evaluate_split(X,train_y,ids,tr,te,truth,models,save_predictions):
    began=time.perf_counter()
    bank=ReductionBank().fit(X[tr],train_y,ids[tr])
    fit_seconds=time.perf_counter()-began
    began=time.perf_counter()
    A=bank.prepare(X[tr]); B=bank.prepare(X[te])
    transform_seconds=time.perf_counter()-began
    scores=[]; predictions={}
    for spec in REDUCERS:
        a=bank.project(A,spec); b=bank.project(B,spec)
        for clf in CLASSIFIERS:
            start=time.perf_counter()
            pipe=build_pipeline(clf,models[clf],preprocess=False).fit(a,train_y)
            fit=time.perf_counter()-start
            start=time.perf_counter()
            pred=pipe.predict(b); prob=pipe.predict_proba(b)[:,1]
            scores.append(dict(reducer=spec['name'],classifier=clf,n_features=a.shape[1],
                               classifier_fit_seconds=fit,predict_seconds=time.perf_counter()-start,
                               **metrics(truth,pred,prob)))
            if save_predictions:
                predictions[spec['name']+'__'+clf]=dict(prediction=pred.tolist(),probability=prob.tolist())
    return dict(scores=scores,predictions=predictions,bank=bank.audit(),
                reduction_metadata={s['name']:bank.metadata(s) for s in REDUCERS},
                bank_fit_seconds=fit_seconds,bank_transform_seconds=transform_seconds)


def choose(rows):
    if not rows: raise ValueError('Empty inner candidate pool')
    return sorted(rows,key=lambda r:(-r['mean_f1_macro'],-r['mean_accuracy'],-r['mean_auc'],
                                     r['mean_features'],r['mode'],r['reducer'],r['classifier']))[0]


def select_from_inner(inner_rows):
    """Selection accepts INNER scores only, never outer predictions or labels."""
    df=pd.DataFrame(inner_rows); ranking=[]
    for (mode,reducer,clf),g in df.groupby(['mode','reducer','classifier'],sort=False):
        if len(g)!=3 or set(g.inner_fold)!={0,1,2}: raise ValueError('Incomplete inner candidate')
        ranking.append(dict(mode=mode,reducer=reducer,classifier=clf,mean_features=float(g.n_features.mean()),
                            **{'mean_'+m:float(g[m].mean()) for m in PRIMARY}))
    selections={}
    for mode in MODES:
        selections[mode+'_raw']=choose([r for r in ranking if r['mode']==mode and r['reducer']=='none'])
        selections[mode+'_compact']=choose([r for r in ranking if r['mode']==mode])
    selections['new_selected']=choose([r for r in ranking if r['mode']!='original'])
    selections['all_selected']=choose(ranking)
    return selections,ranking


def evaluation_signature():
    initialize()
    models=classifiers(42)
    baseline=json.loads((ROOT/'results/cv/settings.json').read_text())
    for c in CLASSIFIERS:
        if json.loads(json.dumps(models[c].get_params(),default=str))!=baseline['classifiers'][c]:
            raise ValueError('Baseline classifier parameters changed')
    config=dict(extraction_settings_sha256=sha(OUT/'settings.json'),
                code_sha256={p:sha(ROOT/p) for p in ['scripts/evaluate_embedding_ablation.py',
                    'src/diagramo/reduction.py','scripts/evaluate.py']},
                original_reference='validated feature optimization; all 18 reducers available in original_compact selection',
                fixed_grid='8 shared reducers x 4 modes x 3 classifiers x 2 branches = 192',
                inner_grid='original 18x3 + new adaptations 3x8x3 = 126 candidates per branch',
                scenarios=[m+'_'+r for m in MODES for r in ('raw','compact')]+['new_selected','all_selected'],
                classifiers={c:models[c].get_params() for c in CLASSIFIERS})
    path=OUT/'evaluation_settings.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(config,default=str)):
        raise ValueError('Evaluation code/configuration changed; cannot reuse saved folds')
    write_json(path,config)
    return sha(path),models


def run_branch(branch,signature,models,workers=2):
    ids,y,folds,splits=load_subjects()
    meta_path=OUT/'embeddings'/f'{branch}.json'
    if not meta_path.exists(): raise RuntimeError('Complete extraction first: '+branch)
    meta=json.loads(meta_path.read_text())
    if meta['settings_sha256']!=sha(OUT/'settings.json'): raise ValueError('Extraction settings mismatch')
    for item in meta['outputs'].values():
        if sha(OUT/'embeddings'/item['file'])!=item['sha256']: raise ValueError('Embedding hash mismatch')
    dest=OUT/'folds'; dest.mkdir(exist_ok=True)
    folds.to_csv(OUT/'fold_assignments.csv',index=False)
    for mode in MODES:
        if mode=='original':
            # Exact original full grid and inner CV retained as comparison/reference.
            for f,(tr,te) in enumerate(splits):
                target=dest/f'{branch}__{mode}__fold_{f}.json'
                if target.exists():
                    old=json.loads(target.read_text())
                    if old['evaluation_sha256']!=signature: raise ValueError('Fold signature mismatch')
                    continue
                source=ROOT/'results/feature_optimization/folds'/f'{branch}__fold_{f}.json'
                old=json.loads(source.read_text(encoding='utf-8'))
                if old['bank']['fit_participants']!=ids[tr].tolist() or old['test_participants']!=ids[te].tolist():
                    raise ValueError('Prior fold mismatch')
                result=dict(branch=branch,mode=mode,fold=f,evaluation_sha256=signature,
                            train_participants=ids[tr].tolist(),test_participants=ids[te].tolist(),
                            test_indices=te.tolist(),bank=old['bank'],scores=old['scores'],
                            predictions=old['predictions'],reduction_metadata=old['reduction_metadata'],
                            inner_scores=old['selection']['inner_fold_scores'],inner_audits=old['selection']['audits'],
                            source_sha256=sha(source),source=str(source.relative_to(ROOT)),
                            bank_fit_seconds=0.,bank_transform_seconds=0.,reused_original=True)
                write_json(target,result)
            continue
        X=np.load(OUT/'embeddings'/meta['outputs'][mode]['file'],mmap_mode='r')
        if len(X)!=len(ids) or not np.isfinite(X).all(): raise ValueError('Invalid embedding matrix')
        def run_one(item):
            f,(tr,te)=item
            target=dest/f'{branch}__{mode}__fold_{f}.json'
            if target.exists():
                if json.loads(target.read_text())['evaluation_sha256']!=signature: raise ValueError('Fold signature mismatch')
                print('REUSE',branch,mode,f,flush=True); return f
            began=time.perf_counter(); print('START',branch,mode,'outer',f+1,flush=True)
            # Inner procedure receives only outer train arrays and labels.
            inner_scores,inner_audits=inner_train_search(X[tr],y[tr],ids[tr],models,branch,mode,f)
            result=evaluate_split(X,y[tr],ids,tr,te,y[te],models,True)
            result.update(branch=branch,mode=mode,fold=f,evaluation_sha256=signature,
                          train_participants=ids[tr].tolist(),test_participants=ids[te].tolist(),
                          test_indices=te.tolist(),inner_scores=inner_scores,inner_audits=inner_audits,
                          seconds=time.perf_counter()-began,reused_original=False)
            write_json(target,result)
            print('COMPLETE',branch,mode,f+1,round(result['seconds'],1),'s',flush=True)
            del result; gc.collect()
            return f
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(run_one,item) for item in enumerate(splits)]
            for future in concurrent.futures.as_completed(futures):
                f=future.result()
                write_json(OUT/'evaluation_progress.json',dict(completed_outer_jobs=len(list(dest.glob('*.json'))),
                           total_outer_jobs=48,last_branch=branch,last_mode=mode,last_fold=f,workers=workers))
        del X; gc.collect()


def inner_train_search(X_train,y_train,ids_train,models,branch,mode,outer_fold):
    scores=[]; audits=[]
    for k,(tr,va) in enumerate(StratifiedKFold(n_splits=3,shuffle=True,random_state=42).split(ids_train,y_train)):
        result=evaluate_split(X_train,y_train[tr],ids_train,tr,va,y_train[va],models,False)
        scores.extend(dict(inner_fold=k,**r) for r in result['scores'])
        audits.append(dict(inner_fold=k,train_participants=ids_train[tr].tolist(),
                           validation_participants=ids_train[va].tolist(),bank=result['bank'],
                           reduction_metadata=result['reduction_metadata'],
                           bank_fit_seconds=result['bank_fit_seconds'],bank_transform_seconds=result['bank_transform_seconds']))
        print('  INNER',branch,mode,outer_fold+1,k+1,'/3',flush=True)
        del result; gc.collect()
    return scores,audits


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',choices=list(BRANCHES),nargs='+',default=list(BRANCHES))
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    args=parser.parse_args()
    signature,models=evaluation_signature()
    with threadpool_limits(limits=2):
        for branch in args.branch: run_branch(branch,signature,models,args.workers)
