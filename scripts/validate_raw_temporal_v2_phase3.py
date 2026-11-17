"""Replay SSL reconstruction and every neural/CPU downstream experiment."""
import subprocess
import joblib
from raw_temporal_v2_common import *
from raw_temporal_v2_training import predict,build_model,SOURCES as TRAIN_SOURCES
from raw_temporal_v2_ssl import evaluate,SOURCES as SSL_SOURCES
from raw_temporal_v2_frozen import SOURCES as FROZEN_SOURCES
from diagramo.temporal_v2.multiscale import MultiScaleLocal,OriginalRetainedLocal
from diagramo.temporal_v2.masked_motor import MaskedMotorModel
from train_raw_temporal_v2_phase3 import CONFIG_PATH,EXTRA


def main():
    seed_all(42);ids,y,schema,train,val,test=development();config=json.loads((ROOT/CONFIG_PATH).read_text());rows=[]
    assert not set(ids[train])&set(ids[val]) and not set(ids[train])&set(ids[test])
    table=pd.read_csv(OUT/'tables/phase3_all_downstream_fits.csv');assert len(table)==50 and table.experiment.nunique()==50
    neural_sig=signature(CONFIG_PATH,TRAIN_SOURCES+EXTRA);cpu_sig=signature(CONFIG_PATH,FROZEN_SOURCES+EXTRA);ssl_sig=signature(CONFIG_PATH,SSL_SOURCES+EXTRA)
    for seed in config['seeds']:
        dest=OUT/'phase3_ssl'/f'p3_masked_ssl_seed{seed}';result=json.loads((dest/'result.json').read_text())
        assert result['complete'] and result['signature']==ssl_sig['signature']
        assert set(result['fit_ids'])==set(ids[train]);assert set(result['validation_ids'])==set(ids[val])
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
        state=torch.load(ROOT/result['checkpoint'],map_location=DEVICE,weights_only=False)
        assert set(state['normalizer'].fit_ids)==set(ids[train])
        local=MultiScaleLocal() if state['local_encoder']=='multiscale' else OriginalRetainedLocal()
        model=MaskedMotorModel(local,4 if state['local_encoder']=='multiscale' else 8).to(DEVICE);model.load_state_dict(state['state'])
        evaluated=evaluate(model,state['normalizer'],ids,val,state['constant'].to(DEVICE),seed+700000)
        expected=pd.read_csv(dest/'validation_reconstruction.csv').set_index('participant_id').loc[ids[val]]
        actual=pd.DataFrame(evaluated).set_index('participant_id').loc[ids[val]]
        error=float(np.abs(expected.model_mse.values-actual.model_mse.values).max());assert error<1e-6
        for key in ('model_mse','train_constant_mse','visible_endpoint_interpolation_mse'):
            np.testing.assert_allclose(expected[key],actual[key],atol=1e-6,rtol=1e-6)
        coverage=pd.read_csv(dest/'train_window_coverage.csv');assert set(coverage.participant_id)==set(ids[train])
        assert (coverage.all_windows==coverage.eligible_windows+coverage.windows_without_pen_down_targets).all()
        rows.append(dict(experiment=result['experiment_id'],kind='masked_ssl',replay_error=error,train_participants=len(train),evaluation_participants=len(val)))
    for row in table.itertuples():
        dest=OUT/'phase3_ssl'/row.experiment;result=json.loads((dest/'result.json').read_text());assert result['complete']
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
        fitids=result['fit_ids'];assert set(fitids)<=set(ids[train]);assert set(result['evaluation_ids'])==set(ids[val])
        expected=pd.read_csv(dest/'evaluation_predictions.csv').set_index('participant_id').loc[ids[val]]
        if 'architecture' in result:
            assert result['signature']==neural_sig['signature']
            state=torch.load(ROOT/result['checkpoint'],map_location=DEVICE,weights_only=False)
            assert set(state['normalizer'].fit_ids)==set(ids[train])
            model=build_model(state['architecture']).to(DEVICE);model.load_state_dict(state['state'])
            logit,z=predict(model,state['normalizer'],ids[val]);probability=1/(1+np.exp(-logit))
            curve=pd.read_csv(dest/'learning_curve.csv');assert np.isfinite(curve.gradient_norm).all()
        else:
            assert result['signature']==cpu_sig['signature']
            state=joblib.load(ROOT/result['checkpoint']);assert set(state['fit_ids'])==set(fitids)
            feature=ROOT/result['feature_path'];assert sha(feature)==result['feature_sha256']
            info=json.loads(feature.with_suffix('.json').read_text());assert not set(info['fingerprint']['fit_ids'])&set(ids[val])
            with np.load(feature) as data:x=data['X'];people=data['ids']
            lookup={pid:i for i,pid in enumerate(people)};a=[lookup[p] for p in fitids];b=[lookup[p] for p in ids[val]]
            np.testing.assert_allclose(state['pipeline'][0].mean_,x[a].astype('float64').mean(0),atol=1e-6,rtol=1e-6)
            assert state['pipeline'][0].n_samples_seen_==len(a)
            probability=state['pipeline'].predict_proba(x[b])[:,1]
        error=float(np.abs(expected.probability.values-probability).max());assert error<1e-6
        for key,value in metrics(y[val],probability).items():
            assert abs(value-result['metrics'][key])<1e-8 and abs(value-getattr(row,key))<1e-8
        rows.append(dict(experiment=row.experiment,kind=row.method,replay_error=error,train_participants=len(fitids),evaluation_participants=len(val)))
        print('PHASE3 VALIDATED',row.experiment,flush=True)
    for seed in (42,43,44):
        small=set(config['labeled_subsets'][f'seed{seed}_fraction0.25']);medium=set(config['labeled_subsets'][f'seed{seed}_fraction0.5'])
        assert small<=medium<=set(ids[train])
    comparison=pd.read_csv(OUT/'tables/phase3_ssl_comparison.csv')
    assert len(comparison)==12
    choices=comparison[comparison.method.isin(['finetune','frozen_ssl_linear','frozen_ssl_nonlinear'])]
    summary=choices.groupby('method')[['f1_macro','auc','trainable_parameters']].mean().reset_index()
    winner=summary.sort_values(['f1_macro','auc','trainable_parameters'],ascending=[False,False,True]).iloc[0]
    selected_hyper=choices[choices.method==winner.method].iloc[0].hyper
    decision=dict(selected_ssl_method=winner.method,development_metrics=winner.to_dict(),
        selected_hyper=None if pd.isna(selected_hyper) else float(selected_hyper),
        architecture=config['architecture'],pretraining=config['pretraining'],supervised_training=config['supervised_training'],
        rule=config['selection'],outer_test_used=False,
        warning='All selection and label sensitivity use reused development cohort; final comparison requires original six-fold nested protocol')
    write(OUT/'phase3_ssl/development_decision.json',decision)
    pd.DataFrame(rows).to_csv(OUT/'tables/phase3_checkpoint_replay.csv',index=False)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_raw_temporal_v2_masked_motor.py','tests/test_raw_temporal_v2_protocol.py','-q'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'phase3_ssl/unit_tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8');assert tests.returncode==0
    budget=json.loads((OUT/'cpu_fit_budget.json').read_text())
    phase_fits={k:v for k,v in budget['subsequent_fits'].items() if k.startswith('p3_')}
    assert len(phase_fits)==38 and all(v['status']=='completed' for v in phase_fits.values())
    write(OUT/'phase3_validation.json',dict(complete=True,masked_ssl_fits=2,neural_downstream_fits=12,cpu_readout_fits=38,
        total_replayed_checkpoints=len(rows),checkpoint_replay_passed=True,unit_tests_passed=True,
        source_signature=neural_sig['signature'],outer_test_loaded=False,normalization_pool_train_only=True,
        label_sensitivity_actual_train_counts=sorted(table.labeled_train_participants.unique().astype(int).tolist()),
        label_sensitivity_fixed_labeled_validation_count=len(val),
        label_sensitivity_not_total_annotation_efficiency=True,phase4_ready=True,selection=decision))
    from reconcile_raw_temporal_v2_ledger import reconcile
    reconcile()


if __name__=='__main__':
    with gpu_worker():main()
