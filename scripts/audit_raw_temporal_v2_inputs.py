"""Train-domain channel fidelity and independent source-consistency diagnosis."""
from raw_temporal_v2_common import *
from diagramo.temporal.preprocessing import CHANNELS, CONTINUOUS, preprocess
from diagramo.temporal.dataset import TrainNormalizer
from diagramo.features import kinematic

CONFIG_PATH='configs/raw_temporal_v2/phase0.json'
CONFIG=json.loads((ROOT/CONFIG_PATH).read_text())
DIRECTORY=OUT/'phase0_diagnostics/source_audit'


def describe(array):
    a=np.asarray(array,dtype='float64');finite=np.isfinite(a);v=a[finite]
    return dict(n=len(a),nonfinite=int((~finite).sum()),zero=int((v==0).sum()),
        minimum=float(v.min()),maximum=float(v.max()),mean=float(v.mean()),std=float(v.std()),
        sum=float(v.sum()),sum_squares=float(np.square(v).sum()))


def independent_derivatives(record,pre):
    t=np.asarray(record['time']);x=np.asarray(record['x']);y=np.asarray(record['y']);pen=np.asarray(record['pen_status'])
    delta=np.diff(t);same=(pen[1:]==pen[:-1]);valid=(delta>=1e-4)&(delta<=1)&same&(pen[1:]==1)
    velocity=np.zeros((len(t),2));at=np.flatnonzero(valid)+1
    velocity[at,0]=(x[at]-x[at-1])/(t[at]-t[at-1]);velocity[at,1]=(y[at]-y[at-1])/(t[at]-t[at-1])
    np.testing.assert_array_equal(pre['velocity_valid'],np.r_[False,valid])
    np.testing.assert_allclose(velocity,pre['physical'][:,8:10],atol=1e-12,rtol=1e-12)
    acceleration=np.zeros_like(velocity);valid_acc=np.r_[False,valid]&np.r_[False,False,valid[:-1]]
    at=np.flatnonzero(valid_acc);acceleration[at]=(velocity[at]-velocity[at-1])/(t[at]-t[at-1])[:,None]
    np.testing.assert_array_equal(pre['acceleration_valid'],valid_acc)
    np.testing.assert_allclose(acceleration,pre['physical'][:,11:13],atol=1e-12,rtol=1e-12)


def protect_sources():
    from raw_temporal_common import protect
    count=protect()
    sources=json.loads((OLD/'audit/source_hashes.json').read_text())
    for item in sources:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    manifest=json.loads((OLD/'reproducibility_manifest.json').read_text())
    checked=0
    for path,digest in {**manifest['source_sha256'],**manifest['artifact_sha256']}.items():
        assert sha(ROOT/path)==digest,path;checked+=1
    return dict(protected_baseline_files=count,original_source_files=len(sources),protected_v1_source_artifacts=checked)


def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    sig=signature(CONFIG_PATH,['scripts/audit_raw_temporal_v2_inputs.py'],['src/diagramo/features.py'])
    begin('p0_source','0','source_audit',sig,hypothesis='Source and preprocessing fidelity',
        split='All sources hash-checked; channel analysis only215 outer Train participants',
        input_representation='Original SVC physical, signed-log, cached float16, Train-normalized clipped',seed=42)
    write(DIRECTORY/'signature.json',sig);protected=protect_sources()
    ids,y,schema,fit,val,test=development();allowed=np.sort(np.r_[fit,val])
    with np.load(CACHE/'statistics.npz') as archive:statistics=archive['statistics']
    norm=TrainNormalizer().fit(statistics,fit,ids[fit])
    write(DIRECTORY/'normalizer.json',dict(fit_ids=norm.fit_ids,mean=norm.mean.tolist(),scale=norm.scale.tolist()))
    rows=[];rawrows=[];angles=[];samples=[];alignment=[];descriptor_rows=[];independent=[]
    selected=set(ids[balanced_tiny(CONFIG['descriptor_comparison_training_participants'])])
    # Three historical source discrepancies are checked separately without labels.
    discrepancies=pd.read_csv(OLD/'audit/raw_svc_comparison.csv')
    special=discrepancies[discrepancies.raw_samples!=discrepancies.merged_samples]
    selected.update(special.participant_id)
    with np.load(ROOT/'data/derived/kinematic.npz') as archive:baseline=archive['X']
    source_features=None;began=time.perf_counter();points=0
    for index in allowed:
        pid=ids[index];records,metadata,path=recording_map(pid,schema['tasks'])
        assert int(metadata['group'])==int(y[index]);role='development_fit' if index in set(fit) else 'development_validation'
        with np.load(CACHE/(pid+'.npz')) as cached:
            for ti,task in enumerate(schema['tasks']):
                record=records[task][1];pre=preprocess(record);independent_derivatives(record,pre)
                physical=pre['physical'];transformed=pre['values'];n=len(transformed);points+=n
                packed=transformed.astype('float16').astype('float32')
                normal=(packed.astype('float64')-norm.mean)/norm.scale;clipped=np.clip(normal,-10,10)
                real=np.arange(128)[None,:]<cached[f'n{ti}'][:,None]
                np.testing.assert_array_equal(cached[f'x{ti}'][real],transformed)
                np.testing.assert_array_equal(cached[f'm{ti}'][:,5],pre['time'][cached[f'm{ti}'][:,0].astype(int)])
                alignment.append(dict(participant_id=pid,task=task,samples=n,exact_cached_float32=True,derivatives_independently_checked=True))
                at=np.unique(np.linspace(0,n-1,min(CONFIG['channel_audit_sample_points_per_recording'],n)).astype(int))
                # Aligned, deterministic stratified samples support quantiles/figures only.
                samples.append(np.stack([physical[at],transformed[at],normal[at],clipped[at]],axis=1))
                for ci,channel in enumerate(CHANNELS):
                    a=describe(physical[:,ci]);b=describe(transformed[:,ci]);c=describe(clipped[:,ci]);error=np.abs(transformed[:,ci]-packed[:,ci])
                    row=dict(participant_id=pid,task=task,role=role,channel=channel,
                        **{'physical_'+k:v for k,v in a.items()},**{'log_'+k:v for k,v in b.items()},
                        **{'normalized_'+k:v for k,v in c.items()},
                        clipping_low=int((normal[:,ci]<-10).sum()),clipping_high=int((normal[:,ci]>10).sum()),
                        quantization_absolute_max=float(error.max()),quantization_absolute_sum=float(error.sum(dtype='float64')))
                    if ci in (8,9,10):row['invalid_derivative_samples']=int((~pre['velocity_valid']).sum())
                    elif ci in (11,12):row['invalid_derivative_samples']=int((~pre['acceleration_valid']).sum())
                    else:row['invalid_derivative_samples']=0
                    rows.append(row)
                for channel,values in record.items():rawrows.append(dict(participant_id=pid,task=task,role=role,channel=channel,**describe(values)))
                azimuth=np.asarray(record['azimuth']);diff=np.diff(azimuth);same=np.asarray(record['pen_status'])[1:]==np.asarray(record['pen_status'])[:-1]
                angles.append(dict(participant_id=pid,task=task,role=role,minimum=float(azimuth.min()),maximum=float(azimuth.max()),
                    jumps_over_180=int((np.abs(diff)>180).sum()),same_pen_state_jumps_over_180=int(((np.abs(diff)>180)&same).sum()),
                    hypothetical_wrapped_difference_max=float(np.abs((diff+180)%360-180).max()),
                    note='360-period diagnostic only; not a verified calibration or automatic unit conversion'))
                if pid in selected:
                    values,names=kinematic(record);source_features=names
                    original=baseline[index,ti*74:(ti+1)*74]
                    for name,value,reference in zip(names,values,original):
                        descriptor_rows.append(dict(participant_id=pid,task=task,feature=name,svc=float(value),protected_cache=float(reference),
                            absolute_difference=float(abs(value-reference)),relative_difference=float(abs(value-reference)/max(abs(reference),1e-8))))
                    keep=np.r_[True,np.diff(np.asarray(record['time']))>0]
                    x=np.asarray(record['x'])[keep];yy=np.asarray(record['y'])[keep];t=np.asarray(record['time'])[keep];p=np.asarray(record['pressure'])[keep]
                    derived={'trajectory_duration':float(t[-1]-t[0]),'trajectory_width':float(x.max()-x.min()),
                        'trajectory_height':float(yy.max()-yy.min()),'trajectory_path_length':float(np.hypot(np.diff(x),np.diff(yy)).sum()),
                        'pressure_mean':float(p.mean()),'pressure_std':float(p.std())}
                    for name,value in derived.items():
                        got=float(values[names.index(name)]);np.testing.assert_allclose(value,got,rtol=2e-7,atol=1e-5)
                        independent.append(dict(participant_id=pid,task=task,feature=name,independently_recomputed=value,shared_extractor=got))
        if len(alignment)%140==0:print('INPUT AUDIT',len(alignment),'recordings',points,'samples',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(DIRECTORY/'channel_by_recording.csv',index=False)
    pd.DataFrame(rawrows).to_csv(DIRECTORY/'raw_sensor_by_recording.csv',index=False)
    pd.DataFrame(angles).to_csv(DIRECTORY/'angular_discontinuities.csv',index=False)
    pd.DataFrame(alignment).to_csv(DIRECTORY/'cache_alignment.csv',index=False)
    descriptors=pd.DataFrame(descriptor_rows);descriptors.to_csv(DIRECTORY/'kinematic_source_comparison.csv',index=False)
    pd.DataFrame(independent).to_csv(DIRECTORY/'independent_descriptors.csv',index=False)
    summary=[];sampled=np.concatenate(samples);quantiles=np.quantile(sampled,[0,.01,.25,.5,.75,.99,1],axis=0)
    for ci,channel in enumerate(CHANNELS):
        d=frame[frame.channel==channel];count=int(d.physical_n.sum());row=dict(channel=channel,n=count,
            clipping_fraction=float((d.clipping_low.sum()+d.clipping_high.sum())/count),
            invalid_derivative_fraction=float(d.invalid_derivative_samples.sum()/count),
            quantization_absolute_max=float(d.quantization_absolute_max.max()),quantization_absolute_mean=float(d.quantization_absolute_sum.sum()/count))
        for stage in ('physical','log','normalized'):
            mean=float(d[stage+'_sum'].sum()/count);row[stage+'_mean']=mean
            row[stage+'_std']=float(np.sqrt(max(0,d[stage+'_sum_squares'].sum()/count-mean**2)))
            row[stage+'_minimum']=float(d[stage+'_minimum'].min());row[stage+'_maximum']=float(d[stage+'_maximum'].max())
            row[stage+'_zero_fraction']=float(d[stage+'_zero'].sum()/count);row[stage+'_nonfinite']=int(d[stage+'_nonfinite'].sum())
        for si,stage in enumerate(('physical','signed_log','normalized_unclipped','normalized_clipped')):
            for qi,name in enumerate(('min','p01','p25','p50','p75','p99','max')):row[stage+'_sample_'+name]=float(quantiles[qi,si,ci])
        summary.append(row)
    pd.DataFrame(summary).to_csv(DIRECTORY/'channel_summary.csv',index=False)
    descriptors.groupby('feature').agg(n=('feature','size'),maximum_absolute_difference=('absolute_difference','max'),
        mean_absolute_difference=('absolute_difference','mean'),maximum_relative_difference=('relative_difference','max')).reset_index().to_csv(DIRECTORY/'descriptor_summary.csv',index=False)
    assert not frame[['physical_nonfinite','log_nonfinite','normalized_nonfinite']].to_numpy().any()
    result=dict(complete=True,signature=sig['signature'],**protected,participants_audited=len(allowed),recordings_audited=len(alignment),
        samples=points,normalizer_fit_participants=len(fit),normalizer_fit_ids=ids[fit].tolist(),excluded_outer_test_ids=ids[test].tolist(),
        channel_order=list(CHANNELS),raw_column_order=['x','y','time','pen_status','azimuth','tilt','pressure'],
        quantile_scope='Deterministic up-to128 stratified sample indices per recording; global sums, clipping and zero counts use every sample',
        no_outer_test_labels_used=True,original_kinematic_unmodified=True,legacy_derivatives_cross_pen_lifts=True,
        raw_derivatives_pen_down_only=True,raw_angular_period_verified=False,seconds=time.perf_counter()-began,
        historical_metadata_issue='Old raw_svc_comparison.json primary_input text still names merged; actual v1 recording_map defaults to SVC, and source/cache checks verify SVC. Preserve that historical file unchanged.')
    write(DIRECTORY/'validation.json',result)
    ledger('p0_source',training_status='completed',completion_status='completed',artifact_paths=[DIRECTORY.relative_to(ROOT).as_posix()],
        validation_metrics=dict(samples=points,records=len(alignment)),training_cost=result['seconds'],
        scientific_interpretation='Integrity checked independently; source/derivative definitions differ from protected summaries and are explicitly documented')
    print('SOURCE INPUT AUDIT COMPLETE',points,flush=True)


if __name__=='__main__':main()
