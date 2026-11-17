"""Cache observable physical targets and exact internal temporal anchors once.

No label is stored or fitted. The original input tensors are reused unchanged.
Reading SVC again is necessary for internal timestamps/physical probe targets
which were not retained in the original patch-boundary-only cache.
"""
import argparse
from raw_temporal_v2_common import *
from diagramo.temporal.preprocessing import preprocess

TARGETS=('log_speed_mean','log_speed_std','log_pressure_std','log_duration',
         'log_path_length','log_width','pen_up_fraction','peak_speed_position')
DIRECTORY=LOCAL/'physical_targets'


def observable(record,pre,start,end):
    speed=pre['physical'][start:end,10];valid=pre['velocity_valid'][start:end].copy();valid[0]=False
    x=np.asarray(record['x'])[start:end];y=np.asarray(record['y'])[start:end]
    pressure=np.asarray(record['pressure'])[start:end];n=end-start
    # The first edge crosses the requested observation's boundary, not its scope.
    length=float(np.hypot(np.diff(x),np.diff(y))[valid[1:]].sum())
    found=speed[valid]
    values=[float(np.log1p(found.mean())) if len(found) else np.nan,
        float(np.log1p(found.std())) if len(found) else np.nan,
        float(np.log1p(pressure.std())),float(np.log1p(pre['time'][end-1]-pre['time'][start])),
        float(np.log1p(length)),float(np.log1p(np.ptp(x))),float((pre['pen'][start:end]==0).mean()),
        float(np.argmax(np.where(valid,speed,-np.inf))/max(n-1,1)) if len(found) else np.nan]
    return values


def statistics_reference(pre,start,end):
    values=pre['values'][start:end].astype('float64')
    return np.r_[values.mean(0),values.std(0)].astype('float32')


def prepare_one(pid,schema,signature_value):
    path=DIRECTORY/(pid+'.npz');meta=DIRECTORY/(pid+'.json')
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text());assert saved['signature']==signature_value
        assert sha(path)==saved['sha256'];return saved
    records,_,_=recording_map(pid,schema['tasks']);window=[];segments=[];tasks=[]
    window_ref=[];segment_ref=[];task_ref=[];tokenmeta=[];patch_duration=[];segment_duration=[];task_duration=[];points=0
    with np.load(CACHE/(pid+'.npz')) as original:
        for ti,task in enumerate(schema['tasks']):
            record=records[task][1];pre=preprocess(record);inventory=original[f'm{ti}'];n=len(pre['time']);points+=n
            for a,b,*rest in inventory:
                a,b=int(a),int(b);window.append(observable(record,pre,a,b));window_ref.append(statistics_reference(pre,a,b))
                patch_duration.append(pre['time'][b-1]-pre['time'][a]);entry=np.zeros((4,4),dtype='float32')
                for j,positions in enumerate(np.array_split(np.arange(17),4)):
                    centers=positions*8;valid=centers-4<(b-a);centers=np.clip(centers[valid],0,b-a-1)
                    if not len(centers):continue
                    first,last=a+int(centers[0]),a+int(centers[-1]);duration=pre['time'][last]-pre['time'][first]
                    values=[pre['time'][first]-pre['time'][0],duration,duration/max(last-first,1),last-first+1]
                    entry[j]=np.log1p(values)/4
                tokenmeta.append(entry)
            for segment in np.unique(inventory[:,2]).astype(int):
                rows=inventory[inventory[:,2]==segment];a,b=int(rows[0,0]),int(rows[-1,1])
                segments.append(observable(record,pre,a,b));segment_ref.append(statistics_reference(pre,a,b))
                segment_duration.append(pre['time'][b-1]-pre['time'][a])
            tasks.append(observable(record,pre,0,n));task_ref.append(statistics_reference(pre,0,n))
            task_duration.append(pre['time'][-1]-pre['time'][0])
    taskvalues=np.asarray(tasks,dtype='float32');taskreferences=np.asarray(task_ref)
    np.savez_compressed(path,window_targets=np.asarray(window,dtype='float32'),segment_targets=np.asarray(segments,dtype='float32'),
        task_targets=taskvalues,person_targets=np.nanmean(taskvalues,axis=0,keepdims=True),
        window_reference=np.asarray(window_ref),segment_reference=np.asarray(segment_ref),task_reference=taskreferences,
        person_reference=taskreferences.mean(0,keepdims=True),token_meta=np.asarray(tokenmeta),
        patch_duration=np.asarray(patch_duration,dtype='float32'),segment_duration=np.asarray(segment_duration,dtype='float32'),
        task_duration=np.asarray(task_duration,dtype='float32'),target_names=np.asarray(TARGETS))
    saved=dict(signature=signature_value,sha256=sha(path),participant_id=pid,raw_samples=points,
        windows=len(window),segments=len(segments),tasks=14,no_clinical_labels=True,
        undefined_motion_targets='NaN where no within-observation valid pen-down interval; never imputed as zero',
        person_scope='Mean of14 observable task-level targets; not chronological concatenation')
    write(meta,saved);return saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args()
    assert json.loads((OUT/'phase0_validation.json').read_text())['engineering_gate_passed']
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    sig=signature('configs/raw_temporal_v2/phase1.json',['scripts/prepare_raw_temporal_v2_targets.py'])
    ids,y,schema,_,_=cohort();records=[]
    # All258 participants receive deterministic, unlabeled metadata. Fitting and
    # all development evaluations still exclude the designated43 outer Test.
    for i,pid in enumerate(ids[:args.limit] if args.limit else ids):
        records.append(prepare_one(pid,schema,sig['signature']))
        if i%10==0:print('PHYSICAL TARGET CACHE',i+1,'participants',flush=True)
    if args.limit is None:
        assert len(records)==258
        write(OUT/'phase1_bottleneck/physical_targets_manifest.json',dict(complete=True,signature=sig,
            participants=258,target_names=list(TARGETS),records=records,no_label_fitting=True,
            source_mapping='Original SVC; original float32 and packed input caches unchanged',
            token_anchor_policy='Four contiguous groups of the17 convolutional map positions; exact source timestamps of clipped receptive-field centres, no uniform-time interpolation',
            coverage='Every original window is represented; short windows can have fewer than4 valid retained tokens'))
    print('PHYSICAL TARGET METADATA FINISHED',len(records),flush=True)


if __name__=='__main__':main()
