"""Replay all final-stage predictions and independently audit fusion scopes."""
import argparse
import hashlib
import joblib
from raw_temporal_v2_common import *
from raw_temporal_v2_evaluation import (config,SOURCES,CONFIG_PATH,probability,logit_features,weight_fit,kinematic_data,key)
from raw_temporal_v2_training import build_model,predict
from diagramo.temporal.dataset import TrainNormalizer
from fusion_raw_temporal_v2 import assert_block


def result(name):return json.loads((OUT/'phase4_complementarity'/name/'result.json').read_text())


def model_validation():
    seed_all(42);cfg=config();ids,y,schema,assignments,splits=cohort();lookup={pid:i for i,pid in enumerate(ids)}
    sig=signature(CONFIG_PATH,SOURCES);rows=[];pretraining={};verified={}
    def register(path):verified[path.relative_to(ROOT).as_posix()]=sha(path)
    with np.load(CACHE/'statistics.npz') as data:statistics=data['statistics']
    for path in sorted((OUT/'phase4_complementarity').glob('*/result.json')):
        r=json.loads(path.read_text());assert r['complete'] and r['signature']==sig['signature']
        checkpoint=ROOT/r['checkpoint'];assert sha(checkpoint)==r['checkpoint_sha256']
        register(path);register(checkpoint)
        fit=np.asarray([lookup[p] for p in r['fit_ids']]);assert 0<len(fit)<258
        if r['experiment_id'].startswith('p4_ssl_pre_'):
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['validation_ids']==[] and r['validation_ids']==[]
            assert set(state['normalizer'].fit_ids)==set(r['fit_ids'])
            expected=TrainNormalizer().fit(statistics,fit,ids[fit]);np.testing.assert_allclose(state['normalizer'].mean,expected.mean,atol=1e-10,rtol=0)
            np.testing.assert_allclose(state['normalizer'].scale,expected.scale,atol=1e-10,rtol=0)
            assert r['epochs_run']==cfg['pretraining_fixed_epochs']
            assert all(torch.isfinite(v).all() for v in state['local_state'].values())
            pretraining[r['checkpoint_sha256']]=set(r['fit_ids'])
            rows.append(dict(experiment=r['experiment_id'],type='pretraining',replay_max_error=0.,
                verification='Checkpoint integrity, finite local state, fixed epochs and Train-only normalizer; downstream predictions replay encoder functionality'))
            continue
        evaluation=np.asarray([lookup[p] for p in r['evaluation_ids']]);assert not set(fit)&set(evaluation)
        expected=pd.read_csv(path.parent/'evaluation_predictions.csv',float_precision='round_trip').set_index('participant_id').loc[ids[evaluation]]
        register(path.parent/'evaluation_predictions.csv')
        if 'architecture' in r:
            runtime=json.loads((path.parent/'fit_config.json').read_text());register(path.parent/'fit_config.json')
            assert runtime['architecture']==r['architecture'] and runtime['seed']==r['seed'] and runtime['fixed_epochs']==r['fixed_epochs']
            assert set(runtime['fit_ids'])==set(r['fit_ids']) and set(runtime['evaluation_ids'])==set(r['evaluation_ids'])
            state=torch.load(checkpoint,map_location=DEVICE,weights_only=False)
            assert set(state['fit_ids'])==set(r['fit_ids']) and set(state['normalizer'].fit_ids)==set(r['fit_ids'])
            normalizer=TrainNormalizer().fit(statistics,fit,ids[fit],index_only=r['architecture'].get('index_only',False))
            np.testing.assert_allclose(state['normalizer'].mean,normalizer.mean,atol=1e-10,rtol=0)
            np.testing.assert_allclose(state['normalizer'].scale,normalizer.scale,atol=1e-10,rtol=0)
            model=build_model(state['architecture']).to(DEVICE);model.load_state_dict(state['state'])
            logits,z=predict(model,state['normalizer'],ids[evaluation]);p=1/(1+np.exp(-logits))
            if r['fixed_epochs'] is not None:assert state['validation_ids']==[] and r['no_evaluation_labels_during_fixed_epoch_training']
        elif 'feature_path' in r:
            state=joblib.load(checkpoint);feature=ROOT/r['feature_path'];assert sha(feature)==r['feature_sha256']
            meta=json.loads(feature.with_suffix('.json').read_text())
            register(feature);register(feature.with_suffix('.json'))
            assert set(meta['fingerprint']['fit_ids'])==set(r['fit_ids'])
            assert not set(meta['fingerprint']['fit_ids'])&set(r['evaluation_ids'])
            with np.load(feature) as data:x=data['X'];people=data['ids']
            position={pid:i for i,pid in enumerate(people)};a=[position[pid] for pid in ids[fit]];b=[position[pid] for pid in ids[evaluation]]
            np.testing.assert_allclose(state['pipeline'][0].mean_,x[a].astype('float64').mean(0),atol=1e-6,rtol=1e-6)
            assert state['pipeline'][0].n_samples_seen_==len(fit)
            p=state['pipeline'].predict_proba(x[b])[:,1]
        elif 'input_sha256' in r:
            state=joblib.load(checkpoint)
            with np.load(path.parent/'cross_fitted_input_provenance.npz') as data:
                np.testing.assert_array_equal(data['train_ids'],ids[fit]);np.testing.assert_array_equal(data['evaluation_ids'],ids[evaluation])
                a=logit_features(data['train_base'][:,r['columns']]);b=logit_features(data['evaluation_base'][:,r['columns']])
            register(path.parent/'cross_fitted_input_provenance.npz')
            digest=hashlib.sha256(a.tobytes()+y[fit].tobytes()+b.tobytes()).hexdigest();assert digest==r['input_sha256']
            assert set(state['fit_ids'])==set(r['fit_ids'])
            np.testing.assert_allclose(state['pipeline'][0].mean_,a.mean(0),atol=1e-10,rtol=1e-10)
            p=state['pipeline'].predict_proba(b)[:,1]
        else:
            model=joblib.load(checkpoint);x=kinematic_data();p=model.predict_proba(x[evaluation])[:,1]
            assert r['preprocessing_train_only_verified'] and model[2].n_samples_seen_==len(fit)
        error=float(np.max(np.abs(p-expected.probability.to_numpy())));assert error<1e-6
        np.testing.assert_array_equal(expected.label,y[evaluation])
        for metric,value in metrics(y[evaluation],p).items():assert abs(value-r['metrics'][metric])<1e-8
        rows.append(dict(experiment=r['experiment_id'],type='downstream_or_combiner',replay_max_error=error,
            verification='Evaluation predictions, metrics and fitted Train scope replayed'))
        print('PHASE4 REPLAY',r['experiment_id'],flush=True)
    # Verify pretrained encoder identity/scope from every downstream link.
    for path in (OUT/'phase4_complementarity').glob('*/result.json'):
        r=json.loads(path.read_text());digest=r.get('pretrained_sha256') or r.get('encoder_checkpoint_sha256')
        if digest:assert digest in pretraining and pretraining[digest]<=set(r['fit_ids'])
    provenance_paths=list((OUT/'phase4_complementarity/fusion_crossfit').glob('*/provenance.json'))
    assert len(provenance_paths)==18
    for path in provenance_paths:
        register(path);register(path.parent/'train_oof.csv')
        proof=json.loads(path.read_text());pool=np.asarray([lookup[pid] for pid in proof['pool_ids']]);seen=[]
        data=pd.read_csv(path.parent/'train_oof.csv',float_precision='round_trip').set_index('participant_id')
        for block in proof['blocks']:
            fit=np.asarray([lookup[pid] for pid in block['fit_ids']]);validation=np.asarray([lookup[pid] for pid in block['validation_ids']])
            kin=result(block['kinematic_experiment']);raw=result(block['raw_experiment'])
            assert_block(pool,fit,validation,kin,raw);seen.extend(validation)
            np.testing.assert_allclose(data.loc[ids[validation],'kinematic_probability'],probability(kin,validation),atol=1e-12)
            np.testing.assert_allclose(data.loc[ids[validation],'raw_probability'],probability(raw,validation),atol=1e-12)
        assert len(seen)==len(pool) and len(set(seen))==len(pool) and set(seen)==set(pool)
    fusion=pd.read_csv(OUT/'phase4_complementarity/fusion_oof.csv',float_precision='round_trip')
    register(OUT/'phase4_complementarity/fusion_oof.csv')
    nested=pd.read_csv(OUT/'phase4_complementarity/nested_oof.csv',float_precision='round_trip')
    register(OUT/'phase4_complementarity/nested_oof.csv')
    expected_models=3 if cfg['preferred_ssl_method'].startswith('frozen_ssl_') else 2
    assert len(nested)==258*2*expected_models and not nested.duplicated(['participant_id','model','seed']).any()
    for (fold,model,seed),group in nested.groupby(['fold','model','seed']):
        train,test=splits[fold];assert group.experiment_id.nunique()==1
        r=result(group.experiment_id.iloc[0]);assert set(r['fit_ids'])==set(ids[train]) and set(r['evaluation_ids'])==set(ids[test])
        expected=group.set_index('participant_id').loc[ids[test]]
        np.testing.assert_allclose(expected.probability,probability(r,test),atol=1e-12,rtol=0)
        assert (expected.checkpoint_sha256==r['checkpoint_sha256']).all()
        np.testing.assert_array_equal(expected.label,y[test]);np.testing.assert_array_equal(expected.prediction,expected.probability>.5)
    for outer,(train,test) in enumerate(splits):
        folder=OUT/'phase4_complementarity'/f'fusion_outer_{outer}'
        provenance=json.loads((folder/'outer_train_crossfit_provenance.json').read_text());seen=[]
        for name in ('outer_train_crossfit_provenance.json','outer_train_oof.csv','combiner.json','fusion_oof.csv'):
            register(folder/name)
        for block in provenance['blocks']:
            fit=np.asarray([lookup[pid] for pid in block['fit_ids']]);validation=np.asarray([lookup[pid] for pid in block['validation_ids']])
            assert_block(train,fit,validation,result(block['kinematic_experiment']),result(block['raw_experiment']));seen.extend(validation)
        assert set(seen)==set(train) and len(seen)==len(train)
        oof=pd.read_csv(folder/'outer_train_oof.csv',float_precision='round_trip').set_index('participant_id').loc[ids[train]]
        matrix=oof[['kinematic_probability','raw_probability']].to_numpy();settings=json.loads((folder/'combiner.json').read_text())
        assert settings['raw_weight']==weight_fit(y[train],matrix)
        for seed,group in fusion[fusion.fold==outer].groupby('seed'):
            frame=group[group.model=='raw_fusion_base'].set_index('participant_id').loc[ids[test]]
            base=np.column_stack([frame.kinematic_probability,frame.probability]);raw=result(frame.raw_experiment.iloc[0])
            assert_block(np.arange(258),train,test,result(f'p4_kin_{key(train)}_{key(test)}'),raw)
            weights=(1-settings['raw_weight'])*base[:,0]+settings['raw_weight']*base[:,1]
            checks={'fixed_average':base.mean(1),'train_selected_weight':weights}
            for method,column in [('stacking','stacking_checkpoint'),('kinematic_calibrated','calibration_checkpoint')]:
                if seed!=42 and method=='kinematic_calibrated':continue
                state=joblib.load(ROOT/settings[column]);assert set(state['fit_ids'])==set(ids[train])
                checks[method]=state['pipeline'].predict_proba(logit_features(base[:,state['columns']]))[:,1]
            for method,p in checks.items():
                expected=group[group.model==method].set_index('participant_id').loc[ids[test]].probability
                np.testing.assert_allclose(expected,p,atol=1e-10,rtol=0)
    pd.DataFrame(rows).to_csv(OUT/'tables/phase4_checkpoint_replay.csv',index=False)
    write(OUT/'phase4_complementarity/model_validation.json',dict(complete=True,source_signature=sig['signature'],
        verified_artifacts=verified,
        replayed_checkpoints=len(rows),fixed_ssl_pretraining_checkpoints=len(pretraining),
        fusion_subinner_crossfit_scopes=len(provenance_paths),all_oof_inputs_participant_disjoint=True,
        early_stopped_inner_predictions_not_reused_for_fusion=True,outer_test_labels_not_used_to_fit_combiners=True))


def main(models_only=False):
    existing=OUT/'phase4_complementarity/model_validation.json'
    if not models_only and existing.exists():
        prior=json.loads(existing.read_text());assert prior['complete'] and prior['source_signature']==signature(CONFIG_PATH,SOURCES)['signature']
        for path,digest in prior['verified_artifacts'].items():assert sha(ROOT/path)==digest,path
    else:model_validation()
    if models_only:return
    for path in ['phase4_complementarity/xai/validation.json','phase4_complementarity/analysis_validation.json']:
        assert json.loads((OUT/path).read_text())['complete'],path
    external=json.loads((OUT/'phase4_complementarity/external/compatibility_audit.json').read_text())
    assert external['status']=='BLOCKED_DATA_NOT_AVAILABLE' and not external['external_metrics_created']
    for fold in range(6):assert json.loads((OUT/'phase4_complementarity'/f'outer_{fold}/nested_fit_status.json').read_text())['fits_complete']
    write(OUT/'phase4_validation.json',dict(complete=True,participants=258,outer_folds=6,inner_folds=3,seed=42,
        checkpoint_replay_passed=True,strict_fusion_cross_fitting_passed=True,
        xai_checked=True,external_evaluation_status=external['status'],historical_adaptive_bias=True,
        inference_scope='Exploratory internal participant CV',delivery_notebook_validation_pending=True))
    from reconcile_raw_temporal_v2_ledger import reconcile
    reconcile()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models-only',action='store_true');args=parser.parse_args()
    with gpu_worker():main(args.models_only)
