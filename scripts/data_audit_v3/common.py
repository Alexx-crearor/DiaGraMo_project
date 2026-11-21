"""Read-only audit primitives. No estimator/checkpoint loading or source writes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/data_audit_v3'
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,f1_score,roc_auc_score,brier_score_loss,log_loss,confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from prepare import TASKS,canonical,loads
SEED=42
RAW_ORDER=('x','y','time','pen_status','azimuth','tilt','pressure')


def sha(path: Path) -> str:
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):raise ValueError('Unsafe input path')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values,dtype='<f8').tobytes()).hexdigest()


def write_json(name: str,value: Any) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv(name: str,frame: pd.DataFrame) -> None:
    OUT.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT/name,index=False)


def read(name: str) -> pd.DataFrame:
    # 'NA' is a real participant prefix, not a missing-value sentinel.
    return pd.read_csv(OUT/name,keep_default_na=False)


def missing(values: pd.Series) -> pd.Series:
    return values.isna()|values.astype(str).str.strip().eq('')


def folds() -> pd.DataFrame:
    frame=pd.read_csv(ROOT/'results/cv/fold_assignments.csv',keep_default_na=False)
    with np.load(ROOT/'data/derived/kinematic.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(frame.participant_id,data['ids']);np.testing.assert_array_equal(frame.label,data['y'])
    if len(frame)!=258 or not frame.participant_id.is_unique:raise ValueError('Invalid protected cohort')
    generated=np.empty(len(frame),dtype=int)
    for number,(_,test) in enumerate(StratifiedKFold(6,shuffle=True,random_state=42).split(frame.participant_id,frame.label)):generated[test]=number
    np.testing.assert_array_equal(frame.fold,generated)
    return frame


def metrics(y: np.ndarray,p: np.ndarray) -> dict[str,Any]:
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if len(y)==0 or y.shape!=p.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid metrics inputs')
    if not set(np.unique(y))<={0,1}:raise ValueError('Nonbinary labels')
    tn,fp,fn,tp=confusion_matrix(y,p>.5,labels=[0,1]).ravel()
    sensitivity=tp/(tp+fn) if tp+fn else np.nan;specificity=tn/(tn+fp) if tn+fp else np.nan
    both=(tp+fn)>0 and (tn+fp)>0
    return dict(n=len(y),group0=int((y==0).sum()),group1=int((y==1).sum()),
        accuracy=float(accuracy_score(y,p>.5)),f1_macro=float(f1_score(y,p>.5,labels=[0,1],average='macro',zero_division=0)),
        auc=float(roc_auc_score(y,p)) if both else np.nan,
        balanced_accuracy=(sensitivity+specificity)/2 if both else np.nan,
        sensitivity=sensitivity,specificity=specificity,brier=float(brier_score_loss(y,p)),
        log_loss=float(log_loss(y,p,labels=[0,1])),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        metric_scope='two_class' if both else 'single_class; macro_F1 fixed labels=[0,1]; AUC/BalAcc undefined')


def md(frame: pd.DataFrame,limit: int=80) -> str:
    def value(item: Any) -> str:
        if item is None or (isinstance(item,str) and item==''):return 'N/A'
        if isinstance(item,(float,np.floating)):return 'N/A' if not np.isfinite(item) else f'{item:.6f}'
        return str(item).replace('|','/').replace('\n',' ')
    shown=frame.head(limit)
    return '\n'.join(['| '+' | '.join(map(str,shown.columns))+' |','| '+' | '.join(['---']*len(shown.columns))+' |',
        *('| '+' | '.join(value(v) for v in row)+' |' for row in shown.itertuples(index=False,name=None))])+('\n\nBảng đầy đủ trong CSV.' if len(frame)>limit else '')


def initialize() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'PROTECTED_BEFORE.csv'
    if target.exists():raise FileExistsError('Audit already initialized; use phases individually without overwriting the baseline snapshot')
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    paths={ROOT/name for name in tracked}
    historical=json.loads((ROOT/'results/raw_temporal/protected_manifest.json').read_text())
    old_manifest=json.loads((ROOT/'results/raw_temporal/reproducibility_manifest.json').read_text())
    historical.update(old_manifest['source_sha256']);historical.update(old_manifest['artifact_sha256'])
    paths.update(ROOT/name for name in historical if (ROOT/name).is_file())
    for folder in ['data','results/raw_temporal/cache','results/models','results/raw_temporal/checkpoints','results/raw_temporal_v2/checkpoints']:
        paths.update(path for path in (ROOT/folder).rglob('*') if path.is_file())
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    paths.update(ROOT/name for name in untracked if not name.startswith(('results/data_audit_v3/','scripts/data_audit_v3/','tests/data_audit_v3/')))
    paths={p for p in paths if p.is_file() and not any(s in p.parts for s in ('__pycache__','.pytest_cache')) and not p.is_relative_to(OUT)}
    rows=[]
    for number,path in enumerate(sorted(paths)):
        rows.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
        if number%1500==0:print('PROTECTED HASHES',number+1,'/',len(paths),flush=True)
    csv('PROTECTED_BEFORE.csv',pd.DataFrame(rows))
    write_json('execution_context.json',dict(reference_commit='35ad4208ae5a8180942a00d8e89a06df3d7ac813',
        actual_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        branch_not_created_reason='Working tree has pre-existing untracked files; conditional clean-tree requirement respected; no stash or checkout',
        preexisting_untracked=[p for p in untracked if not p.startswith(('results/data_audit_v3/','scripts/data_audit_v3/','tests/data_audit_v3/'))],
        seed=42,network_downloads=False,clinical_model_training=False,source_files_read_only=True))
    print('INITIAL SNAPSHOT COMPLETE',len(rows),flush=True)


def verify_protection() -> dict[str,Any]:
    original=read('PROTECTED_BEFORE.csv');bad=[]
    for row in original.itertuples(index=False):
        path=ROOT/row.path
        if not path.exists() or sha(path)!=row.sha256:bad.append(row.path)
    if bad:raise AssertionError(('Protected artifacts changed/missing',bad))
    return dict(complete=True,files_checked=len(original),bytes_checked=int(original.bytes.sum()),changed_files=0)
