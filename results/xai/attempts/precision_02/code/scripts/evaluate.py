"""Fixed six-fold subject-independent evaluation; all transforms fit on training.

Configurations are prespecified (no test-fold tuning). All OOF predictions are saved.
Fusion uses GAF ViT and CWT EfficientNet-B4 fixed from the paper, not local scores.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,itertools,json,time,warnings,sys,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diagramo.kernels import RBFKernelTransformer

def classifiers(seed):
    return {
        'LogisticRegression':LogisticRegression(C=1,max_iter=3000,random_state=seed),
        'SVM':SVC(C=1,kernel='rbf',probability=True,random_state=seed),
        'RandomForest':RandomForestClassifier(n_estimators=300,max_features='sqrt',n_jobs=4,random_state=seed),
        'XGBoost':XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,n_jobs=4,random_state=seed,eval_metric='logloss'),
        'LightGBM':LGBMClassifier(n_estimators=200,num_leaves=15,max_depth=-1,learning_rate=.05,verbosity=-1,n_jobs=4,random_state=seed),
        'AdaBoost':AdaBoostClassifier(n_estimators=100,learning_rate=.5,random_state=seed),
        'MLP':MLPClassifier(hidden_layer_sizes=(128,),alpha=.01,max_iter=500,early_stopping=True,validation_fraction=.15,random_state=seed),
        'Dummy':DummyClassifier(strategy='prior')}

def metrics(y,pred,prob):
    return dict(accuracy=accuracy_score(y,pred),f1_macro=f1_score(y,pred,average='macro',zero_division=0),
                f1_weighted=f1_score(y,pred,average='weighted',zero_division=0),f1_binary=f1_score(y,pred,zero_division=0),
                auc=roc_auc_score(y,prob),balanced_accuracy=balanced_accuracy_score(y,pred))


def build_pipeline(clf_name, clf, *, preprocess=True):
    """Shared fixed classifier pipeline for baseline and fold-local reduction studies.

    Disable preprocessing only after an independently fitted training-fold bank.
    """
    estimator=clone(clf)
    steps=[SimpleImputer(strategy='median'),VarianceThreshold(),StandardScaler()] if preprocess else []
    if clf_name=='SVM' and estimator.kernel=='rbf':
        steps.append(RBFKernelTransformer(gamma=estimator.gamma))
        estimator.set_params(kernel='precomputed')
    return make_pipeline(*steps,estimator)

def summarize(out):
    rows=[]
    for p in out.glob('*__*.json'):
        r=json.loads(p.read_text()); rows.append(dict(representation=r['representation'],classifier=r['classifier'],n_features=r['n_features'],**r['pooled'],**{'mean_'+k:v for k,v in r['mean'].items()},**{'std_'+k:v for k,v in r['std'].items()}))
    pd.DataFrame(rows).sort_values(['accuracy','auc'],ascending=False).to_csv(out/'summary.csv',index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--only',nargs='*'); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--classifiers',nargs='*'); ap.add_argument('--output',default='results/cv'); ap.add_argument('--save-models',action='store_true')
    args=ap.parse_args(); out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/derived'; base=np.load(data/'kinematic.npz'); ids,y=base['ids'],base['y']
    if len(set(ids))!=len(ids): raise ValueError('Subject IDs are not unique')
    arrays={}
    for path in sorted(data.glob('*.npz')):
        d=np.load(path)
        if not np.array_equal(d['ids'],ids) or not np.array_equal(d['y'],y): raise ValueError('Feature rows differ: '+str(path))
        arrays[path.stem]=d['X']
    configs={k:[k] for k in arrays}
    selected={'Kinematic':'kinematic','MOMENT':'moment','GAF':'gaf_vit_b_16','CWT':'cwt_efficientnet_b4'}
    for size in [2,3,4]:
        for combo in itertools.combinations(selected,size):
            keys=[selected[k] for k in combo]
            if all(k in arrays for k in keys): configs['+'.join(combo)]=keys
    if args.only: configs={k:v for k,v in configs.items() if k in args.only}
    if not configs: raise ValueError('No requested feature matrices available')
    splits=list(StratifiedKFold(n_splits=6,shuffle=True,random_state=args.seed).split(ids,y))
    fold_id=np.empty(len(y),dtype=int)
    for f,(tr,te) in enumerate(splits):
        assert not set(ids[tr]) & set(ids[te]); fold_id[te]=f
    pd.DataFrame(dict(participant_id=ids,label=y,fold=fold_id)).to_csv(out/'fold_assignments.csv',index=False)
    models=classifiers(args.seed)
    if args.classifiers: models={k:v for k,v in models.items() if k in args.classifiers}
    settings=dict(seed=args.seed,folds=6,n_subjects=len(y),label='group',positive_label=1,
                  stratified=True,subject_independent=True,f1_primary='macro',
                  fusion_gaf='gaf_vit_b_16',fusion_cwt='cwt_efficientnet_b4',
                  classifiers={k:v.get_params() for k,v in models.items()})
    settings_path=out/'settings.json'
    if settings_path.exists():
        old=json.loads(settings_path.read_text())
        if old['seed']!=args.seed or old['n_subjects']!=len(y): raise ValueError('Changed CV setting: choose a new --output directory')
        for key,value in settings['classifiers'].items():
            if key in old['classifiers'] and old['classifiers'][key]!=json.loads(json.dumps(value,default=str)):
                raise ValueError('Changed classifier parameters: choose a new --output directory')
    (out/'settings.json').write_text(json.dumps(settings,indent=2,default=str),encoding='utf-8')
    def evaluate_configuration(name,keys,classifiers_to_run):
        X=np.concatenate([arrays[k] for k in keys],axis=1)
        if not np.isfinite(X).all(): raise ValueError('Nonfinite feature matrix: '+name)
        for clf_name,clf in classifiers_to_run:
            target=out/(name+'__'+clf_name+'.json')
            if target.exists(): continue
            pred=np.empty(len(y),dtype=int); prob=np.empty(len(y)); fold_results=[]; began=time.time()
            def fit_fold(item):
                f,(tr,te)=item
                # Scaling and variance filtering occur strictly inside each training fold.
                pipe=build_pipeline(clf_name,clf)
                with threadpool_limits(limits=4): pipe.fit(X[tr],y[tr])
                fold_pred=pipe.predict(X[te]); fold_prob=pipe.predict_proba(X[te])[:,1]
                if args.save_models:
                    p=ROOT/'results/models'/name/clf_name; p.mkdir(parents=True,exist_ok=True); joblib.dump(pipe,p/f'fold_{f}.joblib',compress=3)
                return te,fold_pred,fold_prob,dict(fold=f,n_train=len(tr),n_test=len(te),**metrics(y[te],fold_pred,fold_prob))
            # Folds are independent; parallelism changes scheduling, not the split or estimator.
            workers=6 if clf_name=='AdaBoost' else 3
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                for te,fold_pred,fold_prob,fold_result in executor.map(fit_fold,enumerate(splits)):
                    pred[te]=fold_pred; prob[te]=fold_prob; fold_results.append(fold_result)
            oof=pd.DataFrame(dict(participant_id=ids,label=y,fold=fold_id,prediction=pred,probability=prob))
            oof.to_csv(out/(name+'__'+clf_name+'.oof.csv'),index=False)
            result=dict(representation=name,feature_keys=keys,classifier=clf_name,n_features=X.shape[1],
                        pooled=metrics(y,pred,prob),folds=fold_results,
                        mean={k:float(np.mean([r[k] for r in fold_results])) for k in metrics(y,pred,prob)},
                        std={k:float(np.std([r[k] for r in fold_results],ddof=1)) for k in metrics(y,pred,prob)},seconds=time.time()-began)
            target.write_text(json.dumps(result,indent=2),encoding='utf-8')
            summarize(out)
            print(f'{name} / {clf_name}: acc={result["pooled"]["accuracy"]:.4f}, F1macro={result["pooled"]["f1_macro"]:.4f}, AUC={result["pooled"]["auc"]:.4f}, {result["seconds"]:.1f}s',flush=True)
    # Prioritize the published table cells; every other pair remains in the full run.
    priority=[('kinematic','AdaBoost'),('kinematic','LightGBM'),('kinematic','XGBoost'),
              ('moment','SVM'),('moment','LogisticRegression'),('moment','RandomForest'),
              ('Kinematic+MOMENT+GAF','AdaBoost'),('Kinematic+GAF','AdaBoost'),('Kinematic+MOMENT+GAF+CWT','AdaBoost'),
              ('gaf_vit_b_16','LightGBM'),('cwt_efficientnet_b4','LightGBM'),
              ('gaf_efficientnet_b4','XGBoost'),('gaf_vgg16','RandomForest'),('cwt_vit_b_16','SVM'),('cwt_vgg16','SVM')]
    jobs=[(name,clf_name) for name in configs for clf_name in models]
    jobs.sort(key=lambda pair:priority.index(pair) if pair in priority else len(priority))
    for name,clf_name in jobs: evaluate_configuration(name,configs[name],[(clf_name,models[clf_name])])
    summarize(out)
    print('Saved',out/'summary.csv',flush=True)

if __name__=='__main__': main()
