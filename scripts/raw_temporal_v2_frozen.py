"""Frozen local-encoder readouts with deterministic task-level statistics.

No learned aggregator is called a linear probe. Four local tokens are summarized
by mean/max/std for each of the14 tasks, yielding a fixed1344-D participant vector.
Every fit of a CPU estimator consumes the separate CPU-fit resource budget.
"""
import time
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from raw_temporal_v2_common import *
from diagramo.temporal.dataset import load_person
from diagramo.temporal_v2.multiscale import MultiScaleLocal,OriginalRetainedLocal

SOURCES=['scripts/raw_temporal_v2_frozen.py','src/diagramo/temporal_v2/multiscale.py','src/diagramo/temporal_v2/models.py']


def cpu_budget(experiment,sig,status):
    path=OUT/'cpu_fit_budget.json'
    if path.exists():record=json.loads(path.read_text())
    else:
        base=json.loads((OUT/'phase1_bottleneck/capacity_probes/validation.json').read_text())['total_phase1_cpu_probe_fits']
        record=dict(phase1_completed_fits=base,subsequent_fits={})
    previous=record['subsequent_fits'].get(experiment)
    if previous:assert previous['signature']==sig['signature']
    else:
        assert record['phase1_completed_fits']+len(record['subsequent_fits'])<RESOURCE['maximum_cpu_probe_fits'],'CPU fit budget exhausted'
    record['subsequent_fits'][experiment]=dict(signature=sig['signature'],status=status)
    write(path,record)


def features(experiment,checkpoint,train,evaluation,random=False,seed=42):
    began=time.perf_counter()
    seed_all(seed);ids,y,schema,folds,splits=cohort();train=np.asarray(train);evaluation=np.asarray(evaluation)
    assert not set(train)&set(evaluation)
    state=torch.load(checkpoint,map_location=DEVICE,weights_only=False)
    assert set(state['fit_ids'])==set(ids[train]) and set(state['normalizer'].fit_ids)==set(ids[train])
    path=LOCAL/'frozen_features'/f'{experiment}.npz';meta=path.with_suffix('.json');path.parent.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(checkpoint_sha256=sha(checkpoint),random_encoder=random,seed=seed,
        fit_ids=ids[train].tolist(),evaluation_ids=ids[evaluation].tolist(),
        code={p:sha(ROOT/p) for p in SOURCES},aggregation='Within-task mean/max/std of all valid retained local tokens, concatenated actual14 task order')
    if path.exists():
        saved=json.loads(meta.read_text());assert saved['fingerprint']==fingerprint and sha(path)==saved['sha256'];return path
    local=MultiScaleLocal() if state['local_encoder']=='multiscale' else OriginalRetainedLocal()
    if not random:local.load_state_dict(state['local_state'])
    local.to(DEVICE).eval();vectors=[];people=[]
    with torch.no_grad():
        for index in np.r_[train,evaluation]:
            pid=ids[index];item=state['normalizer'].transform(load_person(str(CACHE),pid),DEVICE,'cnn');parts=[];presence=[]
            for start in range(0,len(item['length']),128):
                tokens,valid=local(item['x'][start:start+128],item['length'][start:start+128]);parts.append(tokens);presence.append(valid)
            tokens=torch.cat(parts);valid=torch.cat(presence);taskvectors=[]
            for task in range(14):
                mask=valid&(item['patch_task']==task)[:,None];selected=tokens[mask];assert len(selected)>0
                taskvectors.extend([selected.mean(0),selected.amax(0),selected.std(0,unbiased=False)])
            vectors.append(torch.cat(taskvectors).cpu().numpy());people.append(pid)
    x=np.asarray(vectors);assert x.shape==(len(train)+len(evaluation),1344) and np.isfinite(x).all()
    np.savez_compressed(path,ids=np.asarray(people),X=x)
    write(meta,dict(fingerprint=fingerprint,sha256=sha(path),parameters=sum(p.numel() for p in local.parameters()),
        feature_extraction_seconds=time.perf_counter()-began))
    return path


def classifier(kind,hyper,seed):
    if kind=='linear':model=LogisticRegression(C=hyper,max_iter=3000,random_state=seed)
    else:
        assert kind=='nonlinear'
        model=MLPClassifier(hidden_layer_sizes=(16,),activation='tanh',alpha=hyper,max_iter=300,
            early_stopping=False,n_iter_no_change=20,random_state=seed)
    return make_pipeline(StandardScaler(),model)


def fit_readout(experiment,config_path,feature_path,train,evaluation,kind,hyper,seed=42,extra_sources=(),phase='3'):
    ids,y,schema,folds,splits=cohort();train=np.asarray(train);evaluation=np.asarray(evaluation);assert not set(train)&set(evaluation)
    feature_meta=json.loads(feature_path.with_suffix('.json').read_text());fingerprint=feature_meta['fingerprint']
    assert set(ids[train])<=set(fingerprint['fit_ids'])
    assert not set(fingerprint['fit_ids'])&set(ids[evaluation]),'Encoder was trained on evaluation participants'
    assert sha(feature_path)==feature_meta['sha256']
    assert str(phase) in ('3','4')
    sig=signature(config_path,SOURCES+list(extra_sources));dest=OUT/('phase3_ssl' if str(phase)=='3' else 'phase4_complementarity')/experiment;dest.mkdir(parents=True,exist_ok=True)
    result_path=dest/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['signature']==sig['signature']
        assert result['kind']==kind and result['hyper']==hyper and result['seed']==seed
        assert set(result['fit_ids'])==set(ids[train]) and set(result['evaluation_ids'])==set(ids[evaluation])
        assert result['feature_sha256']==feature_meta['sha256'];assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256'];return result
    cpu_budget(experiment,sig,'running')
    # CPU statuses intentionally use the separate CPU budget; the existing
    # frozen GPU scheduler counts running/completed/failed neural experiments.
    ledger(experiment,phase=str(phase),hypothesis='H3 frozen encoder downstream utility',model='frozen_'+kind,
        input_representation='1344-D fixed statistics of learned local tokens; no learned aggregation',
        ablation=dict(random_encoder=fingerprint['random_encoder'],hyper=hyper),
        split=dict(fit=ids[train].tolist(),evaluation=ids[evaluation].tolist()),seed=seed,
        source_code_hash=sig['source_code_hash'],configuration_hash=sig['configuration_hash'],
        training_status='cpu_running',completion_status='in_progress')
    with np.load(feature_path) as data:x=data['X'];people=data['ids']
    lookup={pid:i for i,pid in enumerate(people)};a=np.asarray([lookup[p] for p in ids[train]]);b=np.asarray([lookup[p] for p in ids[evaluation]])
    model=classifier(kind,hyper,seed);start=time.perf_counter()
    with threadpool_limits(limits=RESOURCE['cpu_threads']):
        model.fit(x[a],y[train]);p=model.predict_proba(x[b])[:,1]
    checkpoint=LOCAL/f'phase{phase}'/f'{experiment}.joblib';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(pipeline=model,fit_ids=ids[train].tolist(),evaluation_ids=ids[evaluation].tolist(),
        feature_sha256=feature_meta['sha256'],signature=sig['signature']),checkpoint)
    loaded=joblib.load(checkpoint);np.testing.assert_allclose(loaded['pipeline'].predict_proba(x[b])[:,1],p,atol=1e-10)
    estimator=model[-1]
    parameter_count=(estimator.coef_.size+estimator.intercept_.size) if kind=='linear' else sum(v.size for v in estimator.coefs_+estimator.intercepts_)
    pd.DataFrame(dict(participant_id=ids[evaluation],label=y[evaluation],probability=p)).to_csv(dest/'evaluation_predictions.csv',index=False)
    result=dict(complete=True,experiment_id=experiment,signature=sig['signature'],seed=seed,kind=kind,hyper=hyper,
        feature_sha256=feature_meta['sha256'],encoder_checkpoint_sha256=fingerprint['checkpoint_sha256'],
        feature_path=feature_path.relative_to(ROOT).as_posix(),parameters=int(parameter_count),encoder_parameters=feature_meta['parameters'],
        random_encoder=fingerprint['random_encoder'],metrics=metrics(y[evaluation],p),dimensions=x.shape[1],
        fit_ids=ids[train].tolist(),evaluation_ids=ids[evaluation].tolist(),training_seconds=time.perf_counter()-start,
        checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint),checkpoint_replay_passed=True,
        feature_scaler_fit_count=model[0].n_samples_seen_.item() if hasattr(model[0].n_samples_seen_,'item') else int(model[0].n_samples_seen_))
    write(result_path,result);cpu_budget(experiment,sig,'completed')
    ledger(experiment,training_status='cpu_completed',completion_status='completed',best_checkpoint=result['checkpoint'],
        validation_metrics=result['metrics'],training_cost=result['training_seconds'],artifact_paths=dest.relative_to(ROOT).as_posix(),
        scientific_interpretation='Fixed encoder downstream readout; clinical label fits use Train participants only')
    print('FROZEN READOUT',experiment,result['metrics'],flush=True);return result
