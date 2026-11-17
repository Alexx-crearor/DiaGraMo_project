"""Frozen final-evaluation services and auditable reusable base fits."""
import hashlib
import time
import joblib
from functools import lru_cache
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from evaluate import classifiers,build_pipeline
from raw_temporal_v2_common import *
from raw_temporal_v2_training import fit as neural_fit,SOURCES as TRAIN_SOURCES
from raw_temporal_v2_ssl import pretrain,SOURCES as SSL_SOURCES
from raw_temporal_v2_frozen import features,fit_readout,SOURCES as FROZEN_SOURCES

CONFIG_PATH='configs/raw_temporal_v2/phase4.json'
SOURCES=list(dict.fromkeys(TRAIN_SOURCES+SSL_SOURCES+FROZEN_SOURCES+[
    'scripts/raw_temporal_v2_evaluation.py','scripts/configure_raw_temporal_v2_phase4.py',
    'scripts/train_raw_temporal_v2_phase4.py','scripts/fusion_raw_temporal_v2.py',
    'scripts/evaluate.py','src/diagramo/kernels.py',
    'results/raw_temporal_v2/phase3_validation.json','results/raw_temporal_v2/phase3_ssl/development_decision.json',
    'data/derived/kinematic.npz','results/cv/kinematic__LightGBM.oof.csv','results/cv/settings.json']))


@lru_cache(maxsize=1)
def config():return json.loads((ROOT/CONFIG_PATH).read_text())


def key(indices):
    ids,_,_,_,_=cohort()
    return hashlib.sha256('|'.join(ids[np.sort(indices)]).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def kinematic_data():
    ids,y,_,_,_=cohort()
    with np.load(ROOT/'data/derived/kinematic.npz') as data:
        np.testing.assert_array_equal(data['ids'],ids);np.testing.assert_array_equal(data['y'],y);x=data['X'].copy()
    assert x.shape==(258,1036) and np.isfinite(x).all()
    return x


@lru_cache(maxsize=1)
def protected_kinematic_models():
    ids,y,schema,folds,splits=cohort();inventory={}
    protected=json.loads((OLD/'protected_manifest.json').read_text())
    for fold,(train,test) in enumerate(splits):
        path=ROOT/'results/models/kinematic/LightGBM'/f'fold_{fold}.joblib'
        relative=path.relative_to(ROOT).as_posix()
        if path.exists() and relative in protected:
            assert sha(path)==protected[relative]
            inventory[key(train)]=dict(path=relative,sha256=sha(path),train_ids=ids[train].tolist(),outer_fold=fold)
        provenance=ROOT/'results/kinematic_xai/train_only'/f'outer_fold_{fold}/provenance.json'
        if not provenance.exists():continue
        prior=json.loads(provenance.read_text());assert set(prior['outer_train_ids'])==set(ids[train])
        for split in prior['inner_splits']:
            path=ROOT/split['checkpoint']
            if not path.exists():continue
            assert sha(path)==split['checkpoint_sha256']
            lookup={pid:i for i,pid in enumerate(ids)};indices=np.asarray([lookup[pid] for pid in split['fit_ids']])
            inventory[key(indices)]=dict(path=split['checkpoint'],sha256=split['checkpoint_sha256'],
                train_ids=split['fit_ids'],outer_fold=None)
    return inventory


def other_cpu_budget(experiment,status):
    path=OUT/'phase4_complementarity/cpu_classification_fit_budget.json'
    record=json.loads(path.read_text()) if path.exists() else dict(fits={},limit=config()['other_cpu_maximum_fits'])
    if experiment not in record['fits']:assert len(record['fits'])<record['limit'],'Other CPU classification budget exhausted'
    record['fits'][experiment]=status;write(path,record)


def kinematic(train,evaluation):
    train=np.sort(train);evaluation=np.sort(evaluation);assert not set(train)&set(evaluation)
    ids,y,schema,folds,splits=cohort();x=kinematic_data();experiment=f'p4_kin_{key(train)}_{key(evaluation)}'
    dest=OUT/'phase4_complementarity'/experiment;dest.mkdir(parents=True,exist_ok=True);target=dest/'result.json'
    sig=signature(CONFIG_PATH,SOURCES)
    if target.exists():
        result=json.loads(target.read_text());assert result['signature']==sig['signature']
        assert set(result['fit_ids'])==set(ids[train]) and set(result['evaluation_ids'])==set(ids[evaluation])
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256'];return result
    source=protected_kinematic_models().get(key(train));began=time.perf_counter()
    if source is not None:
        assert set(source['train_ids'])==set(ids[train]);path=ROOT/source['path'];model=joblib.load(path)
    else:
        other_cpu_budget(experiment,'running')
        with threadpool_limits(limits=RESOURCE['cpu_threads']):model=build_pipeline('LightGBM',classifiers(42)['LightGBM']).fit(x[train],y[train])
        path=LOCAL/'phase4'/(experiment+'.joblib');path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path)
        other_cpu_budget(experiment,'completed')
    # Independently verify fitted preprocessing reflects exactly the Train rows.
    transformed=model[0].transform(x[train]);support=model[1].get_support()
    np.testing.assert_allclose(model[0].statistics_,np.median(x[train],axis=0),atol=1e-7,rtol=1e-8)
    np.testing.assert_allclose(model[2].mean_,transformed[:,support].mean(0),atol=1e-7,rtol=1e-8)
    assert model[2].n_samples_seen_==len(train)
    p=model.predict_proba(x[evaluation])[:,1]
    for fold,(outer_train,outer_test) in enumerate(splits):
        if np.array_equal(train,outer_train):
            old=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv',float_precision='round_trip').set_index('participant_id')
            np.testing.assert_allclose(p,old.loc[ids[evaluation],'probability'],atol=1e-12,rtol=0)
    pd.DataFrame(dict(participant_id=ids[evaluation],label=y[evaluation],probability=p)).to_csv(dest/'evaluation_predictions.csv',index=False)
    result=dict(complete=True,experiment_id=experiment,signature=sig['signature'],fit_ids=ids[train].tolist(),
        evaluation_ids=ids[evaluation].tolist(),checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),
        reused_original_checkpoint=source is not None,metrics=metrics(y[evaluation],p),seconds=time.perf_counter()-began,
        preprocessing_train_only_verified=True)
    write(target,result)
    ledger(experiment,phase='4',model='protected_kinematic_lightgbm',seed=42,
        source_code_hash=sig['source_code_hash'],configuration_hash=sig['configuration_hash'],
        training_status='cpu_reused' if source else 'cpu_completed',completion_status='completed',
        split=dict(fit=ids[train].tolist(),evaluation=ids[evaluation].tolist()),
        best_checkpoint=result['checkpoint'],held_out_metrics=result['metrics'],training_cost=result['seconds'])
    return result


def probability(result,evaluation):
    ids,_,_,_,_=cohort();path=OUT/'phase4_complementarity'/result['experiment_id']/'evaluation_predictions.csv'
    frame=pd.read_csv(path,float_precision='round_trip').set_index('participant_id')
    return frame.loc[ids[evaluation],'probability'].to_numpy()


def ssl_encoder(train,seed):
    cfg=config()
    return pretrain(f'p4_ssl_pre_{key(train)}_seed{seed}',CONFIG_PATH,cfg['pretraining'],np.sort(train),
        fixed_epochs=cfg['pretraining_fixed_epochs'],seed=seed,extra_sources=SOURCES,phase='4')


def neural(experiment,train,evaluation,learning_rate,seed=42,fixed_epochs=None,ssl=False):
    cfg=config();settings=dict(cfg['supervised_training']);settings['learning_rate']=learning_rate
    checkpoint=ROOT/ssl_encoder(train,seed)['checkpoint'] if ssl else None
    ids,_,_,_,_=cohort()
    runtime=dict(architecture=cfg['architecture'],supervised_training=settings,fixed_epochs=fixed_epochs,
        seed=seed,fit_ids=ids[np.sort(train)].tolist(),evaluation_ids=ids[np.sort(evaluation)].tolist(),
        ssl=ssl,pretrained_checkpoint_sha256=sha(checkpoint) if checkpoint else None)
    runtime_path=OUT/'phase4_complementarity'/experiment/'fit_config.json'
    if runtime_path.exists():assert json.loads(runtime_path.read_text())==runtime
    else:write(runtime_path,runtime)
    return neural_fit(experiment,'4',CONFIG_PATH,settings,cfg['architecture'],np.sort(train),
        validation=np.sort(evaluation) if fixed_epochs is None else None,
        evaluation=np.sort(evaluation),fixed_epochs=fixed_epochs,seed=seed,extra_sources=SOURCES,pretrained_local=checkpoint)


def frozen_ssl(train,evaluation,kind,hyper,seed=42,random=False):
    train=np.sort(train);evaluation=np.sort(evaluation);encoder=ssl_encoder(train,seed)
    tag='random' if random else 'ssl'
    feature=features(f'p4_{tag}_features_{key(train)}_{key(evaluation)}_seed{seed}',ROOT/encoder['checkpoint'],train,evaluation,seed=seed,random=random)
    return fit_readout(f'p4_{tag}_{kind}_{key(train)}_{key(evaluation)}_hyper{hyper}_seed{seed}',CONFIG_PATH,
        feature,train,evaluation,kind,hyper,seed=seed,extra_sources=SOURCES,phase='4')


def fixed_raw(train,evaluation,seed=42):
    cfg=config();ssl=cfg['fusion']['source']=='ssl';method=cfg['preferred_ssl_method']
    if ssl and method.startswith('frozen_ssl_'):
        return frozen_ssl(train,evaluation,method.removeprefix('frozen_ssl_'),cfg['fusion']['frozen_hyper'],seed)
    name=f'p4_fusion_base_{key(train)}_{key(evaluation)}_seed{seed}'
    return neural(name,train,evaluation,cfg['fusion']['raw_learning_rate'],seed,
        fixed_epochs=cfg['fusion']['raw_fixed_epochs'],ssl=ssl)


def logit_features(probabilities):
    p=np.clip(np.asarray(probabilities,dtype='float64'),1e-6,1-1e-6)
    return np.log(p/(1-p))


def weight_fit(y,base):
    candidates=[]
    for weight in config()['fusion']['weight_grid']:
        p=(1-weight)*base[:,0]+weight*base[:,1];score=metrics(y,p)
        candidates.append((score['f1_macro'],score['auc'],-weight,weight))
    return max(candidates)[-1]


def meta_fit(experiment,train,evaluation,train_base,evaluation_base,kinematic_only=False):
    train=np.asarray(train);evaluation=np.asarray(evaluation)
    a=np.argsort(train);b=np.argsort(evaluation)
    train_base=np.asarray(train_base)[a];evaluation_base=np.asarray(evaluation_base)[b]
    train=train[a];evaluation=evaluation[b];assert not set(train)&set(evaluation)
    assert train_base.shape==(len(train),2) and evaluation_base.shape==(len(evaluation),2)
    assert np.isfinite(train_base).all() and np.isfinite(evaluation_base).all()
    assert ((train_base>=0)&(train_base<=1)).all() and ((evaluation_base>=0)&(evaluation_base<=1)).all()
    ids,y,_,_,_=cohort();columns=[0] if kinematic_only else [0,1]
    train_features=logit_features(train_base[:,columns]);evaluation_features=logit_features(evaluation_base[:,columns])
    digest=hashlib.sha256(train_features.tobytes()+y[train].tobytes()+evaluation_features.tobytes()).hexdigest()
    sig=signature(CONFIG_PATH,SOURCES);dest=OUT/'phase4_complementarity'/experiment;dest.mkdir(parents=True,exist_ok=True)
    result_path=dest/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['signature']==sig['signature'] and result['input_sha256']==digest
        assert set(result['fit_ids'])==set(ids[train]) and set(result['evaluation_ids'])==set(ids[evaluation]) and result['columns']==columns
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256'];return result
    other_cpu_budget(experiment,'running')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=config()['fusion']['meta_C'],max_iter=3000,random_state=42))
    began=time.perf_counter()
    with threadpool_limits(limits=RESOURCE['cpu_threads']):
        model.fit(train_features,y[train]);p=model.predict_proba(evaluation_features)[:,1]
    path=LOCAL/'phase4'/(experiment+'.joblib');path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(pipeline=model,fit_ids=ids[train].tolist(),evaluation_ids=ids[evaluation].tolist(),
        input_sha256=digest,columns=columns,signature=sig['signature']),path)
    pd.DataFrame(dict(participant_id=ids[evaluation],label=y[evaluation],probability=p)).to_csv(dest/'evaluation_predictions.csv',index=False)
    np.savez_compressed(dest/'cross_fitted_input_provenance.npz',train_ids=ids[train],evaluation_ids=ids[evaluation],
        train_base=train_base,evaluation_base=evaluation_base)
    result=dict(complete=True,experiment_id=experiment,signature=sig['signature'],input_sha256=digest,
        fit_ids=ids[train].tolist(),evaluation_ids=ids[evaluation].tolist(),columns=columns,
        checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),metrics=metrics(y[evaluation],p),
        seconds=time.perf_counter()-began,coefficient=model[-1].coef_.tolist(),intercept=model[-1].intercept_.tolist())
    write(result_path,result);other_cpu_budget(experiment,'completed')
    ledger(experiment,phase='4',model='kinematic_meta_calibration' if kinematic_only else 'regularized_stacking',seed=42,
        source_code_hash=sig['source_code_hash'],configuration_hash=sig['configuration_hash'],
        training_status='cpu_completed',completion_status='completed',best_checkpoint=result['checkpoint'],
        held_out_metrics=result['metrics'],training_cost=result['seconds'],
        scientific_interpretation='Combiner inputs must be Train-only cross-fitted base predictions, never in-sample base predictions')
    return result
