"""Exhaustive original-SVC quality and cached-preprocessing fidelity audit, CPU only."""
from __future__ import annotations
import io
import time
from typing import Any
from scipy.stats import kruskal,mannwhitneyu
from common import *
from diagramo.temporal.preprocessing import preprocess,CHANNELS,CONTINUOUS
from diagramo.temporal.segmentation import windows,padded

FEATURES=['median_dt_positive','estimated_rate_per_source_time','duplicate_dt_fraction','large_gap_fraction',
    'samples','duration_source_time','pressure_mean','pressure_std','pressure_min_observed_step',
    'azimuth_mean','azimuth_std','tilt_mean','tilt_std','x_range','y_range','pen_down_fraction']


def quality(a: np.ndarray) -> dict[str,Any]:
    if a.ndim!=2 or a.shape[1]!=7 or len(a)<1:raise ValueError('SVC must have seven numeric columns')
    n=len(a);dt=np.diff(a[:,2]);positive=dt[dt>0];valid=(dt>=1e-4)&(dt<=1)
    median=float(np.median(positive)) if len(positive) else np.nan
    row=dict(samples=n,nonfinite=int((~np.isfinite(a)).sum()),duration_source_time=float(a[-1,2]-a[0,2]),
        median_dt_positive=median,estimated_rate_per_source_time=1/median if median>0 else np.nan,
        intervals=len(dt),duplicate_dt=int((dt==0).sum()),negative_dt=int((dt<0).sum()),
        tiny_positive_dt=int(((dt>0)&(dt<1e-4)).sum()),large_gaps=int((dt>1).sum()),
        duplicate_dt_fraction=float((dt==0).mean()) if len(dt) else np.nan,
        invalid_dt_fraction=float((~valid).mean()) if len(dt) else np.nan,
        large_gap_fraction=float((dt>1).mean()) if len(dt) else np.nan,
        pen_down_fraction=float((a[:,3]==1).mean()),pen_states='|'.join(map(str,np.unique(a[:,3]))),
        pen_down_valid_duration=float(dt[valid&(a[:-1,3]==1)].sum()),pen_up_valid_duration=float(dt[valid&(a[:-1,3]==0)].sum()),
        positive_interval_sum=float(positive.sum()),x_range=float(np.ptp(a[:,0])),y_range=float(np.ptp(a[:,1])))
    for col,channel in enumerate(RAW_ORDER):
        values=a[:,col];unique=np.unique(values);differences=np.diff(unique);steps=differences[differences>0]
        row.update({channel+'_min':float(values.min()),channel+'_max':float(values.max()),channel+'_mean':float(values.mean()),
            channel+'_std':float(values.std()),channel+'_median':float(np.median(values)),channel+'_p05':float(np.quantile(values,.05)),
            channel+'_p95':float(np.quantile(values,.95)),channel+'_unique':len(unique),
            channel+'_min_observed_step':float(steps.min()) if len(steps) else np.nan,
            channel+'_integer_fraction':float(np.isclose(values,np.rint(values),rtol=0,atol=1e-8).mean())})
    return row


def fdr(p: np.ndarray) -> np.ndarray:
    p=np.asarray(p,dtype=float);finite=np.isfinite(p);out=np.full_like(p,np.nan)
    if not finite.all():out[finite]=fdr(p[finite]);return out
    if len(p)==0:return out
    order=np.argsort(p);ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    adjusted=np.minimum.accumulate(ranked[::-1])[::-1].clip(0,1);out[order]=adjusted;return out


def main() -> None:
    started=time.perf_counter();cohort=read('audit_cohort.csv').set_index('participant_id');selected=set(cohort.index)
    inventory=read('merged_recording_inventory.csv');mapping=inventory.set_index('raw_source')
    if not mapping.index.is_unique:raise AssertionError('Duplicate source recording mapping')
    people=read('participants_inventory.csv').set_index('participant_id')
    people['group']=pd.to_numeric(people['group'],errors='coerce')
    cache=ROOT/'results/raw_temporal/cache';packed_info=json.loads((cache/'packed.json').read_text());bank=np.load(cache/'packed.npy',mmap_mode='r',allow_pickle=False)
    with np.load(cache/'statistics.npz',allow_pickle=False) as data:stats=data['statistics'].copy();ids=data['ids'].copy()
    assignment=folds();np.testing.assert_array_equal(ids,assignment.participant_id)
    means=[];scales=[]
    for fold in range(6):
        train=assignment.fold.to_numpy()!=fold;mean=stats[train,0].mean(0);second=stats[train,1].mean(0)
        scale=np.maximum(np.sqrt(np.maximum(second-mean**2,0)),.1);flags=np.setdiff1d(np.arange(21),CONTINUOUS)
        mean[flags]=0;scale[flags]=1;means.append(mean.astype('float32'));scales.append(scale.astype('float32'))
    rows=[];fidelity=[];comparisons=[];stats_rows=[];missing_sources=[]
    cache_manifests={item['participant_id']:item for item in json.loads((cache/'manifest.json').read_text())}
    rawpaths=sorted((ROOT/'data/raw_data/handwriting_data/data_SVC').rglob('*.svc'))
    by_person={}
    for path in rawpaths:by_person.setdefault(path.parent.name,[]).append(path)
    for person_number,(pid,paths) in enumerate(sorted(by_person.items())):
        is_selected=pid in selected;archive=np.load(cache/(pid+'.npz'),allow_pickle=False) if is_selected else None
        totals=np.zeros((4,21),dtype='float64');total_n=0;local_offset=packed_info['offsets'][pid][0] if is_selected else 0
        # Packed windows follow official task order, not file lexicographic order.
        offsets={}
        if archive is not None:
            for ti,task in enumerate(TASKS):offsets[task]=local_offset;local_offset+=len(archive[f'n{ti}'])
        for path in paths:
            relative=path.relative_to(ROOT).as_posix();rawbytes=path.read_bytes();header=int(rawbytes.split(b'\n',1)[0].strip())
            a=np.loadtxt(io.BytesIO(rawbytes),skiprows=1,ndmin=2)
            task=canonical(path.stem+'.json');source=mapping.loc[relative] if relative in mapping.index else None
            row=dict(participant_id=pid,task=task,selected=is_selected,raw_source=relative,header_samples=header,header_matches=header==len(a),
                **quality(a),**{field:(people.loc[pid,field] if pid in people.index else '') for field in ['id_city','id_school_ppp','group']})
            if not row['header_matches'] or row['nonfinite'] or not set(np.unique(a[:,3]))<={0,1}:raise AssertionError(('Invalid SVC',relative))
            rows.append(row)
            compare=dict(participant_id=pid,task=task,selected=is_selected,merged_mapping_available=source is not None,raw_samples=len(a))
            if source is not None:
                compare['merged_samples']=int(source.samples);compare['sample_count_equal']=len(a)==source.samples
                for ci,channel in enumerate(RAW_ORDER):compare[channel+'_exact']=array_sha(a[:,ci])==source[channel+'_sha256']
                compare['pressure_round4_equal']=array_sha(np.round(a[:,6],4))==source['pressure_round4_sha256']
            comparisons.append(compare)
            if not is_selected:continue
            if task not in TASKS or source is None:raise AssertionError('Selected SVC has no official task mapping')
            ti=TASKS.index(task);record={channel:a[:,ci] for ci,channel in enumerate(RAW_ORDER)}
            pre=preprocess(record);inv=windows(pre);x,ix,length=padded(pre,inv)
            # Independent validity/boundary calculation, not only a self-comparison.
            delta=np.r_[0,np.diff(a[:,2])];dt_valid=(delta>=1e-4)&(delta<=1);dt_valid[0]=False
            pen_change=np.r_[True,np.diff(a[:,3])!=0];boundary=pen_change|~dt_valid
            np.testing.assert_array_equal(pre['boundary'],boundary)
            vv=dt_valid&~pen_change&(a[:,3]==1);np.testing.assert_array_equal(pre['velocity_valid'],vv)
            np.testing.assert_array_equal(pre['acceleration_valid'],vv&np.r_[False,vv[:-1]])
            np.testing.assert_array_equal(archive[f'n{ti}'],length);np.testing.assert_array_equal(archive[f'm{ti}'],inv)
            np.testing.assert_array_equal(archive[f'x{ti}'],x);np.testing.assert_array_equal(archive[f'i{ti}'],ix)
            offset=offsets[task];half=bank[0,offset:offset+len(length)];ihalf=bank[1,offset:offset+len(length)]
            np.testing.assert_array_equal(half,x.astype('float16'));np.testing.assert_array_equal(ihalf,ix.astype('float16'))
            valid=np.arange(128)[None,:]<length[:,None];values=x[valid].astype('float64');index_values=ix[valid].astype('float64')
            total_n+=len(values)
            for j,items in enumerate([values,index_values]):totals[2*j]+=items.sum(0);totals[2*j+1]+=(items**2).sum(0)
            fold=int(cohort.loc[pid,'fold']);packed_values=half[valid].astype('float32');z=(packed_values-means[fold])/scales[fold]
            original_z=(x[valid]-means[fold])/scales[fold];clipped=np.abs(z)>10
            expected=cache_manifests[pid]['raw_source_sha256'][task]
            if hashlib.sha256(rawbytes).hexdigest()!=expected:raise AssertionError('Raw SVC no longer matches cache provenance')
            fidelity.append(dict(participant_id=pid,task=task,fold=fold,n_samples=len(a),windows=len(inv),segments=int(pre['segment'].max()+1),
                source_hash_matches_manifest=True,float32_cache_exact=True,index_cache_exact=True,packed_float16_exact=True,
                independent_masks_exact=True,sample_coverage_exact=int(length.sum())==len(a),
                signed_log_quantization_max=float(np.abs(values-packed_values).max()),normalized_quantization_max=float(np.abs(z-original_z).max()),
                clipping_fraction=float(clipped.mean()),continuous_clipping_fraction=float(clipped[:,CONTINUOUS].mean()),
                clipped_values=int(clipped.sum()),valid_velocity_fraction=float(vv.mean()),boundary_fraction=float(boundary.mean()),
                normalization_fit_scope='Other five outer folds, 215 people; no participant/Test data used in means/SD',
                id_city=cohort.loc[pid,'id_city'],group=int(cohort.loc[pid,'label'])))
        if archive is not None:
            archive.close();index=int(np.flatnonzero(ids==pid)[0]);recomputed=totals/total_n
            np.testing.assert_allclose(recomputed,stats[index],rtol=1e-10,atol=1e-10)
            stats_rows.append(dict(participant_id=pid,samples=total_n,max_statistics_difference=float(np.abs(recomputed-stats[index]).max()),statistics_match=True))
        if person_number%20==0:print('PHASE D ORIGINAL SVC',person_number+1,'/',len(by_person),'participants;',len(rows),'recordings',flush=True)
    for row in inventory.itertuples(index=False):
        if not (ROOT/row.raw_source).is_file():missing_sources.append(row.raw_source)
    if any(path for path in missing_sources if mapping.loc[path,'included']):raise AssertionError('Original SVC missing in selected cohort')
    frame=pd.DataFrame(rows);fidelity=pd.DataFrame(fidelity)
    if fidelity.participant_id.nunique()!=len(selected) or len(fidelity)!=len(selected)*len(TASKS):raise AssertionError('Incomplete cached fidelity audit')
    csv('sensor_quality_by_recording.csv',frame);csv('svc_merged_comparison.csv',pd.DataFrame(comparisons));csv('preprocessing_fidelity.csv',fidelity)
    csv('normalization_statistics_replay.csv',pd.DataFrame(stats_rows))
    numeric=frame.select_dtypes(include=[np.number]).columns.tolist();numeric=[c for c in numeric if c not in ['group']]
    participant=frame.groupby('participant_id')[numeric].mean().reset_index().merge(people.reset_index()[['participant_id','included','id_city','id_school_ppp','group']],on='participant_id',validate='one_to_one')
    counts=frame.groupby('participant_id').size();participant['recordings']=participant.participant_id.map(counts)
    csv('sensor_quality_by_participant.csv',participant)
    analysis=participant[participant.included].copy()
    source_summaries=[]
    for fields in [['id_city'],['id_school_ppp'],['group'],['id_city','group']]:
        for keys,part in analysis.groupby(fields):
            keys=keys if isinstance(keys,tuple) else (keys,)
            for feature in FEATURES:
                v=pd.to_numeric(part[feature],errors='coerce').dropna()
                source_summaries.append(dict(stratification=' × '.join(fields),stratum='|'.join(map(str,keys)),feature=feature,
                    n_participants=len(part),n_finite=len(v),mean=float(v.mean()),median=float(v.median()),p25=float(v.quantile(.25)),p75=float(v.quantile(.75)),
                    unit='source coordinate/pressure/angle/time units or dimensionless, by feature; equal-recording mean within participant'))
    csv('sensor_quality_by_source.csv',pd.DataFrame(source_summaries))
    task_rows=[]
    selected_frame=frame[frame.selected]
    for fields in [['task'],['task','id_city'],['task','group']]:
        for keys,part in selected_frame.groupby(fields):
            keys=keys if isinstance(keys,tuple) else (keys,)
            for feature in FEATURES:task_rows.append(dict(stratification=' × '.join(fields),stratum='|'.join(map(str,keys)),feature=feature,
                n_recordings=len(part),n_participants=part.participant_id.nunique(),mean=float(part[feature].mean()),median=float(part[feature].median())))
    csv('sensor_quality_by_task.csv',pd.DataFrame(task_rows))
    tests=[]
    for feature in FEATURES:
        values=pd.to_numeric(analysis[feature],errors='coerce');parts=[values[analysis.id_city==city].dropna().to_numpy() for city in sorted(analysis.id_city.unique())]
        parts=[part for part in parts if len(part)]
        stat,p=((kruskal(*parts) if np.ptp(np.concatenate(parts))>0 else (0.,1.)) if len(parts)>=2 else (np.nan,np.nan))
        tests.append(dict(feature=feature,contrast='Across cities; source and group confounded',test='Kruskal-Wallis',n=len(analysis),statistic=float(stat),p=float(p),effect=np.nan,
            effect_definition='No single ordered-source effect; descriptive group medians provided'))
        for label,mask in [('group0_vs_group1',np.ones(len(analysis),bool)),('within_city1_group0_vs_group1',analysis.id_city.eq('city1').to_numpy())]:
            a=values[mask&analysis.group.eq(0)].dropna().to_numpy();b=values[mask&analysis.group.eq(1)].dropna().to_numpy()
            stat,p=mannwhitneyu(a,b,alternative='two-sided',method='asymptotic') if len(a) and len(b) else (np.nan,np.nan)
            tests.append(dict(feature=feature,contrast=label,test='Mann-Whitney U',n=len(a)+len(b),n0=len(a),n1=len(b),statistic=float(stat),p=float(p),
                effect=float(2*stat/(len(a)*len(b))-1) if len(a) and len(b) else np.nan,effect_definition='Rank-biserial: positive means larger group0; not causal; city1 positives scarce'))
    tests=pd.DataFrame(tests);tests['fdr_bh_global']=fdr(tests.p.to_numpy());csv('acquisition_signature_analysis.csv',tests)
    csv('preprocessing_by_source.csv',fidelity.groupby(['id_city','group'])[['clipping_fraction','continuous_clipping_fraction','signed_log_quantization_max','normalized_quantization_max','boundary_fraction','valid_velocity_fraction']].agg(['mean','median','max']).reset_index().pipe(lambda d:d.set_axis(['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in d.columns],axis=1)))
    write_json('D_STATUS.json',dict(complete=True,raw_recordings=len(frame),raw_participants=frame.participant_id.nunique(),
        selected_recordings=len(fidelity),selected_participants=fidelity.participant_id.nunique(),all_masks_and_caches_match=True,
        statistical_unit='participant; recording-level summaries descriptive only',device_calibration_verified=False,
        missing_original_files=missing_sources,hardware_causation_identified=False,elapsed_seconds=time.perf_counter()-started))
    print('PHASE D COMPLETE',round(time.perf_counter()-started,2),'seconds',flush=True)


if __name__=='__main__':main()
