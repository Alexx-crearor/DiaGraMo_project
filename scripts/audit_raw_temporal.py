"""Audit every original chronological recording; no resampling or row removal."""
import time
import numpy as np
import pandas as pd
from raw_temporal_common import *

def main():
    began=time.perf_counter();ids,y,schema,folds,splits=cohort();rows=[];runs=[];sources=[];totals={}
    for pi,pid in enumerate(ids):
        records,meta,path=recording_map(pid,schema['tasks']);assert meta['group']==int(y[pi]);sources.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
        for task in schema['tasks']:
            key,r=records[task];a={c:np.asarray(r[c],dtype=np.float64)for c in schema['channels']}
            n=len(a['time']);assert all(len(z)==n for z in a.values());t=a['time'];dt=np.diff(t);positive=dt[dt>0];pen=a['pen_status'];cuts=np.r_[0,np.flatnonzero(pen[1:]!=pen[:-1])+1,n]
            row=dict(participant_id=pid,task=task,source_key=key,n_samples=n,duration_endpoint=float(t[-1]-t[0]),
              duplicate_dt=int((dt==0).sum()),negative_dt=int((dt<0).sum()),tiny_positive_dt=int(((dt>0)&(dt<1e-4)).sum()),
              large_dt_over_1s=int((dt>1).sum()),dt_min=float(dt.min()),dt_max=float(dt.max()),dt_positive_median=float(np.median(positive)),
              dt_positive_p01=float(np.quantile(positive,.01)),dt_positive_p99=float(np.quantile(positive,.99)),
              dt_positive_cv=float(positive.std()/positive.mean()),effective_hz=float((n-1)/(t[-1]-t[0])),median_positive_hz=float(1/np.median(positive)),
              elapsed_positive_sum=float(np.maximum(dt,0).sum()),pen_down_samples=int((pen==1).sum()),pen_up_samples=int((pen==0).sum()),
              pen_states='|'.join(map(str,np.unique(pen))),segments=len(cuts)-1,strokes=int(sum(pen[s]==1 for s in cuts[:-1])),
              missing_channels=0,nonfinite=sum(int((~np.isfinite(z)).sum())for z in a.values()),
              float32_raw_bytes=n*7*4)
            for c in ('x','y','pressure','tilt','azimuth'):row[c+'_min']=float(a[c].min());row[c+'_max']=float(a[c].max())
            rows.append(row)
            source=raw_svc_path(pid,key);sources.append(dict(path=source.relative_to(ROOT).as_posix(),sha256=sha(source)))
            for si,(start,end)in enumerate(zip(cuts[:-1],cuts[1:])):
                runs.append(dict(participant_id=pid,task=task,run_index=si,pen_state=int(pen[start]),raw_start=int(start),raw_end_exclusive=int(end),n_samples=int(end-start),
                    time_start=float(t[start]),time_last=float(t[end-1]),duration_endpoint=float(t[end-1]-t[start]),positive_elapsed=float(np.maximum(np.diff(t[start:end]),0).sum())))
        if pi%30==0:print('AUDITED',pi+1,'/258',flush=True)
    dest=OUT/'audit';dest.mkdir(exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(dest/'recordings.csv',index=False);pd.DataFrame(runs).to_csv(dest/'pen_state_runs.csv',index=False)
    write(dest/'source_hashes.json',sources)
    summary=dict(participants=len(ids),recordings=len(rows),channels=schema['channels'],label='group',class_counts=dict(zip(map(str,np.unique(y)),map(int,np.bincount(y)))),
      samples=int(df.n_samples.sum()),n_samples_quantiles=df.n_samples.quantile([0,.25,.5,.75,.95,.99,1]).to_dict(),duration_quantiles=df.duration_endpoint.quantile([0,.5,.95,.99,1]).to_dict(),
      duplicate_dt=int(df.duplicate_dt.sum()),negative_dt=int(df.negative_dt.sum()),tiny_dt=int(df.tiny_positive_dt.sum()),large_gaps=int(df.large_dt_over_1s.sum()),
      nonfinite=int(df.nonfinite.sum()),pen_states=sorted(df.pen_states.unique()),raw_float32_gib=float(df.float32_raw_bytes.sum()/1024**3),seconds=time.perf_counter()-began,
      timestamp_units='source numerical time, treated as seconds following baseline; no device calibration certificate supplied',
      policy='Keep every sample in recorded acquisition order. No sorting/deduplication. Invalid dt<=0, dt<0.0001 or dt>1 second starts a new derivative/segment boundary. Keep original timestamps, masks and anomaly indicators; do not invent elapsed time across resets.')
    write(dest/'summary.json',summary);write(OUT/'validation.json',dict(complete=False,stage='raw_audit_complete'));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
