"""Fixed cohort/protocol and provenance for raw-signal-only temporal research."""
import hashlib,json,os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,balanced_accuracy_score,recall_score
from prepare import canonical,loads
OUT=ROOT/'results/raw_temporal'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def cohort():
    # Never load the 1,036 handcrafted feature matrix as neural input.
    with np.load(ROOT/'data/derived/kinematic.npz')as d:ids=d['ids'];y=d['y']
    schema=json.loads((ROOT/'data/derived/schema.json').read_text());folds=pd.read_csv(ROOT/'results/cv/fold_assignments.csv')
    assert len(ids)==258 and len(set(ids))==258 and schema['label']=='group'
    np.testing.assert_array_equal(ids,folds.participant_id);np.testing.assert_array_equal(y,folds.label)
    from sklearn.model_selection import StratifiedKFold
    splits=list(StratifiedKFold(6,shuffle=True,random_state=42).split(ids,y))
    for f,(tr,te)in enumerate(splits):np.testing.assert_array_equal(te,np.flatnonzero(folds.fold==f))
    return ids,y,schema,folds,splits
def raw_svc_path(pid,source_key):
    return ROOT/'data/raw_data/handwriting_data/data_SVC'/pid/(source_key[:-5]+'.svc')
def recording_map(pid,tasks,raw_svc=True):
    path=ROOT/'data/clean_data'/(pid+'_merged.json');d=loads(path.read_bytes());records={}
    meta=next(v[0]for k,v in d.items()if k.endswith('_cognitive_test_data')and v)
    for k,v in d.items():
        task=canonical(k)
        if task in tasks:
            assert task not in records;records[task]=(k,v['data'])
    assert set(records)==set(tasks)
    if raw_svc:
        order=('x','y','time','pen_status','azimuth','tilt','pressure')
        for task,(key,_)in records.items():
            source=raw_svc_path(pid,key);A=np.loadtxt(source,skiprows=1)
            with source.open()as f:count=int(f.readline().strip())
            assert A.ndim==2 and A.shape==(count,7),source
            records[task]=(key,{c:A[:,i]for i,c in enumerate(order)})
    return records,meta,path
def metrics(y,p):
    pred=np.asarray(p)>.5
    return dict(f1_macro=f1_score(y,pred,average='macro'),accuracy=accuracy_score(y,pred),auc=roc_auc_score(y,p),
      balanced_accuracy=balanced_accuracy_score(y,pred),sensitivity=recall_score(y,pred,pos_label=1),specificity=recall_score(y,pred,pos_label=0))
def protect():
    m=json.loads((OUT/'protected_manifest.json').read_text())
    for rel,digest in m.items():assert sha(ROOT/rel)==digest,rel
    return len(m)

