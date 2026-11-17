"""Replay Phase 0 results without promoting synthetic scores to real evidence."""
import subprocess
from raw_temporal_v2_common import *
from diagnose_raw_temporal_v2_memorization import predictions
from diagramo.temporal.dataset import load_person
from diagramo.temporal.hierarchical_model import RawTemporalModel
from synthetic_raw_temporal_v2 import fixture,TESTS,ComponentClassifier,tensors,predict


def main():
    seed_all(42);folder=OUT/'phase0_diagnostics'
    source=json.loads((folder/'source_audit/validation.json').read_text());assert source['complete']
    assert source['participants_audited']==215 and source['recordings_audited']==3010
    assert source['normalizer_fit_participants']==143 and source['no_outer_test_labels_used']
    summary=pd.read_csv(folder/'source_audit/channel_summary.csv')
    assert len(summary)==21 and not summary[['physical_nonfinite','log_nonfinite','normalized_nonfinite']].to_numpy().any()
    assert summary.quantization_absolute_max.max()<=.00390625
    ids,y,schema,fit,val,test=development();tiny=[];synthetic=[];source_hashes={}
    for path in sorted(folder.glob('p0_tiny_*/result.json')):
        row=json.loads(path.read_text());assert row['complete'] and row['memorized'] and not row['outer_test_evaluated']
        sig=json.loads((path.parent/'signature.json').read_text())
        for name,digest in sig['source_files'].items():assert sha(ROOT/name)==digest,name;source_hashes[name]=digest
        assert row['fit_ids']==ids[balanced_tiny(row['size'])].tolist()
        assert set(row['fit_ids'])<=set(ids[fit]) and not set(row['fit_ids'])&set(ids[test])
        assert sha(ROOT/row['checkpoint'])==row['checkpoint_sha256']
        checkpoint=torch.load(ROOT/row['checkpoint'],map_location='cpu',weights_only=False)
        model=RawTemporalModel(**checkpoint['architecture']).to(DEVICE);model.load_state_dict(checkpoint['state']);model.eval()
        normalizer=checkpoint['normalizer'];assert normalizer.fit_ids==row['fit_ids']
        tensor=[normalizer.transform(load_person(str(CACHE),pid),DEVICE,row['family']) for pid in row['fit_ids']]
        logit,_=predictions(model,tensor);saved=pd.read_csv(path.parent/'training_predictions.csv')
        np.testing.assert_allclose(logit,saved.logit,atol=1e-4,rtol=0)
        for key,value in metrics(saved.label,saved.probability).items():
            if value is not None:np.testing.assert_allclose(value,row['train_metrics'][key],atol=1e-12,rtol=0)
        gradients=pd.read_csv(path.parent/'gradient_updates.csv')
        assert gradients.gradient_nonfinite.sum()==0
        local=gradients[gradients.module=='local'];assert local.gradient_norm.max()>0 and local.parameter_update_norm.max()>0
        assert row['embedding_variance_mean']>0 and row['logit_sd']>0
        assert pd.read_csv(path.parent/'batch_sensitivity.csv').maximum_absolute_logit_difference.max()<1e-4
        tiny.append(row)
        del tensor,model
        if DEVICE=='cuda':torch.cuda.empty_cache()
    assert len(tiny)==6 and {(r['family'],r['size']) for r in tiny}=={(f,s) for f in ('cnn','tcn','hierarchical') for s in (8,16)}
    for path in sorted(folder.glob('p0_synthetic_*/result.json')):
        row=json.loads(path.read_text());assert row['complete'] and sha(ROOT/row['checkpoint'])==row['checkpoint_sha256']
        sig=json.loads((path.parent/'signature.json').read_text())
        for name,digest in sig['source_files'].items():assert sha(ROOT/name)==digest,name;source_hashes[name]=digest
        data=fixture(row['test'],64,row['seed']+3000);saved=pd.read_csv(path.parent/'heldout_predictions.csv')
        np.testing.assert_array_equal(saved.label,data['y'])
        model=ComponentClassifier(row['family']).to(DEVICE)
        checkpoint=torch.load(ROOT/row['checkpoint'],map_location='cpu',weights_only=False)
        assert checkpoint['signature']==row['signature'];model.load_state_dict(checkpoint['state'])
        logit=predict(model,tensors(data));p=1/(1+np.exp(-logit))
        np.testing.assert_allclose(p,saved.probability,atol=3e-6,rtol=0)
        for key,value in metrics(saved.label,saved.probability).items():
            if value is not None:np.testing.assert_allclose(value,row['heldout_metrics'][key],atol=1e-12,rtol=0)
        if row['test']=='event_order':assert row['statistics_baseline']['accuracy']==.5
        synthetic.append(row)
        del model
        if DEVICE=='cuda':torch.cuda.empty_cache()
    assert len(synthetic)==24
    expected={(kind,family,seed) for kind in TESTS for family in ('cnn','tcn','hierarchical') for seed in (42,43)}
    assert {(r['test'],r['family'],r['seed']) for r in synthetic}==expected
    tests=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_raw_temporal_v2.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (folder/'unit_tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8');assert tests.returncode==0
    write(OUT/'phase0_validation.json',dict(complete=True,engineering_gate_passed=True,
        source_audit_passed=True,real_training_diagnostics=6,synthetic_fits=24,tiny_memorization_passed=6,
        real_outer_test_participants_scored=0,gradient_updates_reach_local_encoder=True,
        checkpoint_predictions_replayed=True,unit_tests=6,source_sha256=source_hashes,
        scope='Phase0 only: optimization validity and component competence; not downstream generalization or bottleneck identification',
        unresolved_fundamental_implementation_errors=[],negative_synthetic_results_are_reported=True))
    ledger('p0_synthetic',training_status='completed',completion_status='completed',
        scientific_interpretation='24 known-label synthetic component fits and independent mean/std control; read actual scores before interpretation',
        artifact_paths=['results/raw_temporal_v2/phase0_diagnostics'])
    state=json.loads((OUT/'validation.json').read_text());state.update(complete=False,phase0='complete',phase1='ready',
        unfinished=['Phases1–4 experimental execution, integrated notebook/reports and validation'])
    write(OUT/'validation.json',state)
    print('PHASE0 VALIDATED: source fidelity, six tiny fits,24 synthetic fits, checkpoint replay',flush=True)


if __name__=='__main__':main()
