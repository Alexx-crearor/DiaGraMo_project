"""Feature selection / dimensionality reduction study in the existing project.

Uses identical stored six outer folds and classifiers from evaluate.py. Inner
3-fold CV refits ALL transforms from scratch, selects reduction + classifier on
mean macro-F1, and never receives outer hold-out labels. Old results are read-only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diagramo.reduction import ReductionBank,choose_candidate,reduction_grid
from evaluate import build_pipeline,classifiers,metrics

warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names')

BRANCHES={'Kinematic':'kinematic','MOMENT':'moment','GAF':'gaf_vit_b_16','CWT':'cwt_efficientnet_b4'}
CLASSIFIERS=['LogisticRegression','SVM','LightGBM']
SPECS=[dict(name='none',method='none',value=None),*reduction_grid()]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def run_candidates(bank,train_X,train_y,test_X,test_y,specs,models):
    """Fit each classifier after transforms fitted only on bank.fit_ids_."""
    train=bank.prepare(train_X); test=bank.prepare(test_X)
    scores=[]; predictions={}
    for spec in specs:
        A=bank.project(train,spec); B=bank.project(test,spec)
        for name in CLASSIFIERS:
            began=time.perf_counter()
            pipe=build_pipeline(name,models[name],preprocess=False).fit(A,train_y)
            pred=pipe.predict(B); prob=pipe.predict_proba(B)[:,1]
            row=dict(reducer=spec['name'],classifier=name,n_features=A.shape[1],
                     seconds=time.perf_counter()-began,**metrics(test_y,pred,prob))
            scores.append(row)
            predictions[(spec['name'],name)]=dict(prediction=pred.tolist(),probability=prob.tolist())
    return scores,predictions


def inner_search(X,y,ids,models,outer_fold):
    """Only outer TRAIN data are arguments to the selection procedure."""
    rows=[]; audits=[]
    for inner,(tr,va) in enumerate(StratifiedKFold(n_splits=3,shuffle=True,random_state=42).split(ids,y)):
        bank=ReductionBank().fit(X[tr],y[tr],ids[tr])
        scores,_=run_candidates(bank,X[tr],y[tr],X[va],y[va],SPECS,models)
        for score in scores: rows.append(dict(inner_fold=inner,**score))
        audits.append(dict(inner_fold=inner,train_participants=ids[tr].tolist(),
                           validation_participants=ids[va].tolist(),bank=bank.audit()))
        print(f'  outer {outer_fold}: inner {inner+1}/3 complete',flush=True)
    df=pd.DataFrame(rows)
    ranking=[]
    for (reducer,clf),g in df.groupby(['reducer','classifier'],sort=False):
        ranking.append(dict(reducer=reducer,classifier=clf,mean_features=float(g.n_features.mean()),
                            **{'mean_'+m:float(g[m].mean()) for m in ('accuracy','f1_macro','auc')}))
    return dict(optimized=choose_candidate(ranking),
                baseline=choose_candidate([r for r in ranking if r['reducer']=='none']),
                ranking=ranking,inner_fold_scores=rows,audits=audits)


def load_data():
    arrays={}; ids=y=None
    folds=pd.read_csv(ROOT/'results/cv/fold_assignments.csv')
    for branch,key in BRANCHES.items():
        with np.load(ROOT/'data/derived'/(key+'.npz')) as d:
            X,these_ids,these_y=d['X'],d['ids'].astype(str),d['y']
        if ids is None: ids,y=these_ids,these_y
        if not np.array_equal(these_ids,ids) or not np.array_equal(these_y,y): raise ValueError('Branch alignment differs')
        if not np.isfinite(X).all(): raise ValueError('Nonfinite input features')
        arrays[branch]=X
    if len(set(ids))!=len(ids) or not np.array_equal(folds.participant_id,ids) or not np.array_equal(folds.label,y):
        raise ValueError('Stored participant/label identity mismatch')
    split=list(StratifiedKFold(n_splits=6,shuffle=True,random_state=42).split(ids,y))
    for f,(tr,te) in enumerate(split):
        if set(ids[tr]) & set(ids[te]) or not (folds.fold.iloc[te]==f).all():
            raise ValueError('Outer folds differ from the original baseline')
    return arrays,ids,y,folds,split


def baseline_predictions(key,clf,ids,y,folds):
    path=ROOT/'results/cv'/f'{key}__{clf}.oof.csv'
    d=pd.read_csv(path,float_precision='round_trip')
    if not np.array_equal(d.participant_id,ids) or not np.array_equal(d.label,y) or not np.array_equal(d.fold,folds.fold):
        raise ValueError('Baseline OOF alignment mismatch')
    return d


def run_fold(branch,f,tr,te,X,ids,y,folds,models,out,settings_hash):
    path=out/'folds'/f'{branch}__fold_{f}.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['settings_sha256']!=settings_hash: raise ValueError('Fold settings changed')
        return branch,f,'resumed'
    began=time.perf_counter()
    print(f'START {branch} fold {f+1}/6',flush=True)
    # No full-cohort fit is performed; this bank belongs only to the outer train.
    bank=ReductionBank().fit(X[tr],y[tr],ids[tr])
    scores,predictions=run_candidates(bank,X[tr],y[tr],X[te],y[te],SPECS[1:],models)
    for name in CLASSIFIERS:
        old=baseline_predictions(BRANCHES[branch],name,ids,y,folds).iloc[te]
        pred=old.prediction.to_numpy(); prob=old.probability.to_numpy()
        scores.append(dict(reducer='none',classifier=name,n_features=bank.fitted_features_,seconds=0.,
                           reused_original_oof=True,**metrics(y[te],pred,prob)))
        predictions[('none',name)]=dict(prediction=pred.tolist(),probability=prob.tolist())
    print(f'{branch} fold {f+1}: outer grid complete; selecting on TRAIN only',flush=True)
    selection=inner_search(X[tr],y[tr],ids[tr],models,f'{branch}/{f+1}')
    result=dict(branch=branch,representation=BRANCHES[branch],fold=f,settings_sha256=settings_hash,
                train_participants=ids[tr].tolist(),test_participants=ids[te].tolist(),
                test_indices=te.tolist(),bank=bank.audit(),
                reduction_metadata={s['name']:bank.metadata(s) for s in SPECS},
                scores=scores,predictions={r+'__'+c:v for (r,c),v in predictions.items()},
                selection=selection,seconds=time.perf_counter()-began)
    write_json(path,result)
    print(f'COMPLETE {branch} fold {f+1}/6 ({result["seconds"]:.1f}s); inner choice: '
          f'{selection["optimized"]["reducer"]}/{selection["optimized"]["classifier"]}',flush=True)
    return branch,f,'completed'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--output',default='results/feature_optimization')
    ap.add_argument('--report-only',action='store_true')
    args=ap.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'results') or out==ROOT/'results':
        raise ValueError('Use a dedicated child of results; original results cannot be overwritten')
    out.mkdir(parents=True,exist_ok=True); (out/'folds').mkdir(exist_ok=True)
    if args.report_only:
        from report_feature_optimization import generate_report
        generate_report(out); return
    if args.workers<1 or args.workers>3: raise ValueError('Use 1–3 workers to bound memory/CPU usage')
    arrays,ids,y,folds,splits=load_data(); models=classifiers(42)
    previous_settings=json.loads((ROOT/'results/cv/settings.json').read_text())
    for name in CLASSIFIERS:
        current=json.loads(json.dumps(models[name].get_params(),default=str))
        if current!=previous_settings['classifiers'][name]: raise ValueError('Baseline classifier parameters differ')
    config=dict(schema=1,seed=42,outer_folds=6,inner_folds=3,n_subjects=len(y),
                branches=BRANCHES,classifiers=CLASSIFIERS,reducers=SPECS,
                primary_selection_metric='mean_inner_f1_macro',
                tie_break=['accuracy','auc','fewer_features','reducer_name','classifier_name'],
                pca='full SVD; standardized inputs; no whitening/post-PCA scaling',
                tree_selector=dict(n_estimators=200,min_samples_leaf=2,max_features='sqrt',n_jobs=1,random_state=42),
                versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
                baseline_settings_sha256=sha(ROOT/'results/cv/settings.json'),
                fold_assignments_sha256=sha(ROOT/'results/cv/fold_assignments.csv'),
                input_sha256={key:sha(ROOT/'data/derived'/(key+'.npz')) for key in BRANCHES.values()},
                code_sha256={p:sha(ROOT/p) for p in ('src/diagramo/reduction.py','scripts/optimize_features.py','scripts/evaluate.py')})
    config_path=out/'settings.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=config:
        raise ValueError('Study configuration changed: use another output subdirectory')
    write_json(config_path,config)
    settings_hash=sha(config_path)
    manifest_path=out/'original_artifacts.json'
    if not manifest_path.exists():
        originals={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for folder in ('results','notebooks')
                   for p in (ROOT/folder).rglob('*') if p.is_file() and not p.is_relative_to(out)}
        write_json(manifest_path,originals)
    folds.to_csv(out/'fold_assignments.csv',index=False)
    jobs=[(branch,f,tr,te) for branch in BRANCHES for f,(tr,te) in enumerate(splits)]
    write_json(out/'validation.json',dict(complete=False,expected_outer_jobs=24,
                                        expected_grid_results=len(BRANCHES)*len(CLASSIFIERS)*len(SPECS)))
    try:
        with threadpool_limits(limits=2),concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_fold,b,f,tr,te,arrays[b],ids,y,folds,models,out,settings_hash) for b,f,tr,te in jobs]
            for future in concurrent.futures.as_completed(futures):
                branch,f,status=future.result()
                done=len(list((out/'folds').glob('*.json')))
                write_json(out/'progress.json',dict(completed_outer_jobs=done,total_outer_jobs=24,
                                                   last_branch=branch,last_fold=f,status=status))
    except BaseException as error:
        write_json(out/'error.json',dict(error=str(error)))
        raise
    from report_feature_optimization import generate_report
    generate_report(out)
    print('FEATURE OPTIMIZATION COMPLETE',out,flush=True)


if __name__=='__main__': main()
