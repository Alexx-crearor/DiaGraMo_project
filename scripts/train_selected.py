"""Refit selected classifiers on all subjects for later local feature inference.

These fitted models are not used to estimate performance; use saved OOF scores.
"""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from evaluate import classifiers

def main():
    valid=json.loads((ROOT/'results/validation.json').read_text())
    if not valid['complete']: raise RuntimeError('Finish and validate the full CV matrix first')
    s=pd.read_csv(ROOT/'results/cv/summary.csv'); best=s[s.classifier!='Dummy'].sort_values(['accuracy','auc'],ascending=False).iloc[0]
    chosen=[('best_cv',best.representation,best.classifier),('paper_three_modality','Kinematic+MOMENT+GAF','AdaBoost')]
    out=ROOT/'results/models'; out.mkdir(exist_ok=True); rows=[]
    for label,rep,clf in chosen:
        r=json.loads((ROOT/'results/cv'/(rep+'__'+clf+'.json')).read_text()); keys=r['feature_keys']
        data=[np.load(ROOT/'data/derived'/(k+'.npz')) for k in keys]
        if not all(np.array_equal(d['ids'],data[0]['ids']) and np.array_equal(d['y'],data[0]['y']) for d in data): raise ValueError('Feature alignment mismatch')
        X=np.concatenate([d['X'] for d in data],axis=1); y=data[0]['y']
        pipe=make_pipeline(SimpleImputer(strategy='median'),VarianceThreshold(),StandardScaler(),classifiers(42)[clf])
        with threadpool_limits(limits=4): pipe.fit(X,y)
        bundle=dict(pipeline=pipe,feature_keys=keys,n_features=X.shape[1],label='group',positive_label=1,
                    training_ids=data[0]['ids'],training_size=len(y),seed=42,cv_pooled=r['pooled'],
                    implementation='Independent reconstruction; see EXPERIMENT_PROTOCOL.md',
                    label_definition='Dataset group column; not identical to special_care_type_dysgraphia',
                    weights_manifest='references/weights_manifest.json',
                    note='Fitted on the full cohort; use OOF predictions for evaluation. Inference requires the exact same task order, transforms and frozen encoders.')
        path=out/(label+'.joblib'); joblib.dump(bundle,path,compress=3)
        rows.append(dict(name=label,representation=rep,classifier=clf,n_features=X.shape[1],training_size=len(y),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path.relative_to(ROOT)),cv_pooled=r['pooled']))
        print('FITTED',label,rep,clf,flush=True)
    (out/'selected_models.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__': main()
