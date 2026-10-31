"""Exploratory label/QC sensitivity on the kinematic branch; same held-out IDs."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate import classifiers,metrics
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd

def main():
    d=np.load(ROOT/'data/derived/kinematic.npz'); X,y,ids=d['X'],d['y'],d['ids']
    meta=pd.read_csv(ROOT/'data/derived/participants.csv').set_index('participant_id').loc[ids]
    splits=list(StratifiedKFold(n_splits=6,shuffle=True,random_state=42).split(ids,y)); rows=[]
    out=ROOT/'results/sensitivity'; out.mkdir(exist_ok=True,parents=True)
    scenarios={'group_all':(y,np.ones(len(y),dtype=bool)),
               'diagnosis_same_cohort':(meta.diagnosis_dysgraphia.to_numpy(dtype=int),np.ones(len(y),dtype=bool)),
               'group_without_omit_flags':(y,meta.flag_to_omit_total.to_numpy()==0)}
    for scenario,(target,mask) in scenarios.items():
        for name in ['AdaBoost','LightGBM']:
            pred=np.empty(len(y),dtype=int); prob=np.empty(len(y)); folds=np.empty(len(y),dtype=int)
            for f,(tr,te) in enumerate(splits):
                tr=tr[mask[tr]]; te=te[mask[te]]
                p=make_pipeline(VarianceThreshold(),StandardScaler(),clone(classifiers(42)[name]))
                with threadpool_limits(limits=4): p.fit(X[tr],target[tr])
                pred[te]=p.predict(X[te]); prob[te]=p.predict_proba(X[te])[:,1]; folds[te]=f
            row=dict(scenario=scenario,classifier=name,n=int(mask.sum()),**metrics(target[mask],pred[mask],prob[mask])); rows.append(row)
            pd.DataFrame(dict(participant_id=ids[mask],label=target[mask],fold=folds[mask],prediction=pred[mask],probability=prob[mask])).to_csv(out/(scenario+'__'+name+'.oof.csv'),index=False)
            print(row,flush=True)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)

if __name__=='__main__': main()
