"""Audit clean_data once, select complete subjects, cache original-signal features."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json, re, hashlib, collections
import numpy as np
import pandas as pd
from diagramo.features import CHANNELS, kinematic, resample
try:
    import orjson
    loads=orjson.loads
except ImportError:
    loads=json.loads

TASKS=['RCFT-recognition','RCFT-recall','dictation','big-spiral-fast','small-spiral-fast',
       'saw-fast','upper-loops-fast','lower-loops-fast','rainbow-fast','combined-loops-fast',
       'small-spiral-precise','lower-loops-precise','combined-loops-precise','copy']

def canonical(key):
    m=re.search(r'_TSK(\d+)\(TA\)-(.+)_\d+\.json$',key)
    if not m: return None
    name=m[2]
    if name.startswith('dictation-'): return 'dictation'
    if name.startswith('copy-'): return 'copy'
    return name

def main():
    out=ROOT/'data/derived'; out.mkdir(parents=True,exist_ok=True)
    reports=ROOT/'results/audit'; reports.mkdir(parents=True,exist_ok=True)
    subjects=[]; records=[]; cohort=[]; sequences=[]; features=[]; hashes=[]
    for i,path in enumerate(sorted((ROOT/'data/clean_data').glob('*_merged.json'))):
        raw=path.read_bytes(); d=loads(raw); pid=path.stem.removesuffix('_merged')
        meta=next((v[0] for k,v in d.items() if k.endswith('_cognitive_test_data') and v),{})
        bytask=collections.defaultdict(list); issues=[]
        for key,obj in d.items():
            task=canonical(key)
            if task not in TASKS: continue
            r=obj['data']; lens={k:len(r.get(k,[])) for k in CHANNELS}
            valid=len(set(lens.values()))==1 and min(lens.values())>=4
            nonfinite=sum(int((~np.isfinite(np.asarray(r.get(k,[]),dtype=float))).sum()) for k in CHANNELS)
            t=np.asarray(r.get('time',[]),dtype=float)
            badtime=int((np.diff(t)<=0).sum())
            records.append(dict(participant_id=pid,task=task,source_key=key,n_samples=lens['x'],
                                lengths_equal=len(set(lens.values()))==1,nonfinite=nonfinite,
                                nonincreasing_timestamps=badtime,meta_samples=obj.get('meta_data',{}).get('samples_count')))
            if not valid or nonfinite: issues.append(task+':invalid_signals')
            bytask[task].append(r)
        missing=[k for k in TASKS if k not in bytask]
        duplicate=[k for k,v in bytask.items() if len(v)>1]
        group=meta.get('group'); diagnosis=meta.get('special_care_type_dysgraphia')
        # Dataset grouping is distinct from the narrower dysgraphia diagnosis flag.
        eligible=not missing and not duplicate and not issues and group in [0,1]
        row=dict(participant_id=pid,task_count=len(bytask),group=group,diagnosis_dysgraphia=diagnosis,
                 flag_to_omit_total=meta.get('flag_to_omit_total'),included=eligible,
                 missing_tasks='|'.join(missing),duplicate_tasks='|'.join(duplicate),issues='|'.join(issues))
        subjects.append(row)
        hashes.append(dict(file=path.name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        if eligible:
            feats=[]; seq=[]
            for task in TASKS:
                r=bytask[task][0]; f,names=kinematic(r); feats.append(f)
                seq.append(np.stack([resample(r[k],512) for k in CHANNELS]))
            cohort.append(row); features.append(feats); sequences.append(seq)
        if i%20==0: print(f'Processed {i+1}/294; included {len(cohort)}',flush=True)
    pd.DataFrame(subjects).to_csv(reports/'participants.csv',index=False)
    pd.DataFrame(records).to_csv(reports/'recordings.csv',index=False)
    pd.DataFrame(cohort).to_csv(out/'participants.csv',index=False)
    (reports/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    ids=np.array([r['participant_id'] for r in cohort]); y=np.array([r['group'] for r in cohort],dtype=np.int64)
    np.savez_compressed(out/'kinematic.npz',X=np.asarray(features).reshape(len(y),-1),y=y,ids=ids)
    np.save(out/'signals512.npy',np.asarray(sequences,dtype=np.float32))
    (out/'schema.json').write_text(json.dumps(dict(tasks=TASKS,channels=CHANNELS,features_per_task=names,
        label='group',positive_label=1,warning='group denotes dataset handwriting-disability grouping; not identical to specific dysgraphia flag',
        selected_participants=len(y),paper_participants=258),indent=2),encoding='utf-8')
    summary=dict(files=len(subjects),recordings=len(records),complete_14=sum(r['task_count']==14 for r in subjects),
                 included=len(y),class_counts={str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},
                 nonfinite_samples=int(sum(r['nonfinite'] for r in records)),
                 nonincreasing_timestamps=int(sum(r['nonincreasing_timestamps'] for r in records)))
    (reports/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(summary,flush=True)
    if len(y)!=258: print('WARNING: cohort differs from paper; inspect audit before claiming reproduction.',flush=True)

if __name__=='__main__': main()
