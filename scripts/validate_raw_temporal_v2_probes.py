"""Reload every physical probe and replay predictions and Train scaling."""
import joblib
from probe_raw_temporal_v2 import weights
from raw_temporal_v2_common import *


def replay(info,feature_path,prediction_path):
    assert sha(ROOT/info['checkpoint'])==info['checkpoint_sha256']
    saved=joblib.load(ROOT/info['checkpoint'])
    ids,y,schema,fit,val,test=development();assert set(saved['fit_ids'])==set(ids[fit])
    assert set(saved['validation_ids'])==set(ids[val])
    with np.load(feature_path) as data:x=data['x'];target=data['y'];people=data['participant_id'];names=data['target_names']
    finite=np.isfinite(target).all(1)&np.isfinite(x).all(1);x=x[finite];target=target[finite];people=people[finite]
    train=np.flatnonzero(np.isin(people,ids[fit]));evaluation=np.flatnonzero(np.isin(people,ids[val]))
    assert not set(people)&set(ids[test])
    w=weights(people[train]);np.testing.assert_allclose(saved['x_scaler'].mean_,np.average(x[train],axis=0,weights=w),atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(saved['y_scaler'].mean_,np.average(target[train],axis=0,weights=w),atol=1e-6,rtol=1e-6)
    transformed=saved['x_scaler'].transform(x[evaluation])
    if saved.get('pca') is not None:
        assert saved['pca'].n_samples_==len(train);transformed=saved['pca'].transform(transformed)
    if saved.get('kernel') is not None:transformed=saved['kernel'].transform(transformed)
    prediction=saved['y_scaler'].inverse_transform(saved['model'].predict(transformed))
    expected=pd.read_csv(prediction_path);assert expected.participant_id.tolist()==people[evaluation].tolist()
    old=expected[['pred_'+name for name in names]].values
    error=float(np.max(np.abs(old-prediction)));assert error<1e-5
    np.testing.assert_allclose(expected[['true_'+name for name in names]].values,target[evaluation],atol=1e-6,rtol=1e-6)
    return dict(checkpoint=info['checkpoint'],maximum_replay_error=error,train_scaling_passed=True)


def main(record=False):
    rows=[];directory=OUT/'phase1_bottleneck';manifest=json.loads((directory/'probes/feature_manifest.json').read_text())
    def append(info,features,predictions,group,method,sig):
        rows.append(replay(info,features,predictions))
        if record:
            assert info['signature']==sig['signature']
            ledger('p1_physical_probe_'+group+'_'+method,phase='1',
                hypothesis='Physical information accessibility; frozen encoder, no clinical objective',
                model=method,input_representation=group,ablation='Physical multioutput probe8 targets',
                split='143 development Train /72 participant-held-out validation; inner3 Train-only selection',seed=42,
                source_code_hash=sig['source_code_hash'],configuration_hash=sig['configuration_hash'],
                training_status='cpu_completed',completion_status='completed',best_checkpoint=info['checkpoint'],
                validation_metrics=info['records'],training_cost=dict(cpu_estimator_fits=10),
                artifact_paths=predictions.relative_to(ROOT).as_posix(),
                scientific_interpretation='CPU budget separate from neural fits; probe accuracy is not clinical causality')
    layer_sig=manifest['signature']
    for level,feature in manifest['features'].items():
        assert sha(ROOT/feature['path'])==feature['sha256']
        for method in ('ridge','rbf_ridge'):
            info=json.loads((directory/'probes'/f'{level}_{method}.json').read_text())
            append(info,ROOT/feature['path'],directory/'probes'/f'{level}_{method}_predictions.csv',level,method,layer_sig)
    pool_sig=signature('configs/raw_temporal_v2/phase1.json',['scripts/probe_raw_temporal_v2_pooling.py','scripts/probe_raw_temporal_v2.py','src/diagramo/temporal_v2/models.py'])
    for mode in ('mean','mean_max','mean_max_std','attention','multi4'):
        feature=LOCAL/'pooling_probe_features'/f'{mode}.npz'
        for method in ('ridge','rbf_ridge'):
            info=json.loads((directory/'pooling_probes'/f'{mode}_{method}.json').read_text())
            append(info,feature,directory/'pooling_probes'/f'{mode}_{method}_predictions.csv','pool_'+mode,method,pool_sig)
    capacity_sig=signature('configs/raw_temporal_v2/phase1.json',['scripts/probe_raw_temporal_v2_capacity.py','scripts/probe_raw_temporal_v2.py'])
    for family in ('cnn','hierarchical'):
        feature=ROOT/manifest['features'][family+'_feature_map']['path']
        for method in ('ridge','rbf_ridge'):
            info=json.loads((directory/'capacity_probes'/f'{family}_{method}.json').read_text())
            append(info,feature,directory/'capacity_probes'/f'{family}_{method}_predictions.csv',family+'_pca32',method,capacity_sig)
    write(directory/'probe_replay_validation.json',dict(complete=True,replayed_checkpoints=len(rows),
        train_scaling_passed=True,outer_test_loaded=False,records=rows))
    print('PHYSICAL PROBE CHECKPOINTS REPLAYED',len(rows),flush=True)


if __name__=='__main__':main()
