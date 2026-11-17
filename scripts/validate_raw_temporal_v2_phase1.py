"""Replay every Phase1 checkpoint and audit split/metric consistency."""
import subprocess
from raw_temporal_v2_common import *
from train_raw_temporal_v2_phase1 import CONFIG_PATH,SOURCES,experiments,predict,RecoveryModel
from synthetic_raw_temporal_v2 import fixture,tensors,predict as synthetic_predict
from synthetic_raw_temporal_v2_pooling import PoolingClassifier


def main():
    seed_all(42);sig=signature(CONFIG_PATH,SOURCES);ids,y,schema,fit,val,test=development();rows=[]
    for spec in experiments():
        path=OUT/'phase1_bottleneck'/('p1_'+spec['name'])/'result.json'
        result=json.loads(path.read_text());assert result['complete'] and result['signature']==sig['signature']
        ckpt=ROOT/result['checkpoint'];assert sha(ckpt)==result['checkpoint_sha256']
        saved=torch.load(ckpt,map_location=DEVICE,weights_only=False)
        assert set(saved['fit_ids'])==set(ids[fit]) and set(saved['normalizer'].fit_ids)==set(ids[fit])
        assert set(saved['validation_ids'])==set(ids[val]);assert not set(saved['fit_ids'])&set(ids[test])
        model=RecoveryModel(**saved['architecture']).to(DEVICE);model.load_state_dict(saved['state'])
        logits,z=predict(model,ids[val],saved['normalizer'],True);p=1/(1+np.exp(-logits))
        expected=pd.read_csv(path.parent/'validation_predictions.csv').set_index('participant_id').loc[ids[val]]
        error=float(np.max(np.abs(p-expected.probability.values)));assert error<1e-6
        computed=metrics(y[val],p)
        for key,value in computed.items():assert abs(value-result['metrics'][key])<1e-8
        curve=pd.read_csv(path.parent/'learning_curve.csv');assert np.isfinite(curve.gradient_norm_mean).all()
        assert np.isfinite(z).all() and not result['outer_test_evaluated']
        rows.append(dict(experiment=result['experiment_id'],checkpoint_replay_maximum_error=error,
            train_participants=len(fit),validation_participants=len(val),outer_test_loaded=False))
        print('VALIDATED',result['experiment_id'],flush=True)
        del model
    config=json.loads((ROOT/'configs/raw_temporal_v2/phase1_synthetic.json').read_text())
    synthetic_sig=signature('configs/raw_temporal_v2/phase1_synthetic.json',[
        'scripts/synthetic_raw_temporal_v2_pooling.py','scripts/synthetic_raw_temporal_v2.py','src/diagramo/temporal_v2/models.py'])
    for kind in config['tests']:
        for mode in config['pooling']:
            for seed in config['seeds']:
                eid=f'p1_synthetic_{kind}_{mode}_seed{seed}';path=OUT/'phase1_bottleneck'/eid/'result.json'
                result=json.loads(path.read_text());assert result['complete'] and result['signature']==synthetic_sig['signature']
                assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
                saved=torch.load(ROOT/result['checkpoint'],map_location=DEVICE,weights_only=False)
                model=PoolingClassifier(mode).to(DEVICE);model.load_state_dict(saved['state'])
                raw=fixture(kind,config['test_bags'],seed+3000);data=tensors(raw)
                logits=synthetic_predict(model,data);p=1/(1+np.exp(-logits))
                old=pd.read_csv(path.parent/'heldout_predictions.csv');error=float(np.abs(p-old.probability.values).max());assert error<1e-6
                for key,value in metrics(raw['y'],p).items():assert abs(value-result['heldout_metrics'][key])<1e-8
                rows.append(dict(experiment=eid,checkpoint_replay_maximum_error=error,train_participants=0,validation_participants=0,outer_test_loaded=False))
    probe=json.loads((OUT/'phase1_bottleneck/probes/validation.json').read_text());assert probe['complete'] and not probe['outer_test_loaded']
    pooling_probe=json.loads((OUT/'phase1_bottleneck/pooling_probes/validation.json').read_text())
    assert pooling_probe['complete'] and not pooling_probe['outer_test_loaded']
    capacity_probe=json.loads((OUT/'phase1_bottleneck/capacity_probes/validation.json').read_text())
    assert capacity_probe['complete'] and not capacity_probe['outer_test_loaded']
    from validate_raw_temporal_v2_probes import main as validate_probes
    validate_probes(record=True)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_raw_temporal_v2.py','tests/test_raw_temporal_v2_models.py','-q'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'phase1_bottleneck/unit_tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8');assert tests.returncode==0
    pd.DataFrame(rows).to_csv(OUT/'tables/phase1_checkpoint_replay.csv',index=False)
    write(OUT/'phase1_validation.json',dict(complete=True,real_development_fits=len(experiments()),
        synthetic_fits=len(rows)-len(experiments()),probe_cpu_fits=capacity_probe['total_phase1_cpu_probe_fits'],checkpoint_replay_passed=True,
        unit_tests_passed=True,source_signature=sig['signature'],outer_test_evaluated=False,
        exploratory_development_only=True,phase2_ready=True))


if __name__=='__main__':
    with gpu_worker():main()
