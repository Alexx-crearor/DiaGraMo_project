"""Physical retention probes for the five matched Phase1 local readouts."""
import joblib
from probe_raw_temporal_v2 import fit_probe,weights,sample
from raw_temporal_v2_common import *
from diagramo.temporal.dataset import load_person
from diagramo.temporal_v2.models import RecoveryModel

CONFIG_PATH='configs/raw_temporal_v2/phase1.json'
CONFIG=json.loads((ROOT/CONFIG_PATH).read_text())
DEST=OUT/'phase1_bottleneck/pooling_probes'
FEATURES=LOCAL/'pooling_probe_features'


def main():
    seed_all(42);torch.set_num_threads(2);DEST.mkdir(parents=True,exist_ok=True);FEATURES.mkdir(parents=True,exist_ok=True)
    ids,labels,schema,train,valid,test=development();fitids=ids[train];valids=ids[valid]
    sig=signature(CONFIG_PATH,['scripts/probe_raw_temporal_v2_pooling.py','scripts/probe_raw_temporal_v2.py','src/diagramo/temporal_v2/models.py'])
    folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(fitids,labels[train]));records=[];coverage=[];replays=[]
    for mode in CONFIG['local_pooling']:
        result=json.loads((OUT/'phase1_bottleneck'/f'p1_cnn_local_{mode}/result.json').read_text())
        checkpoint=ROOT/result['checkpoint'];assert sha(checkpoint)==result['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert set(saved['normalizer'].fit_ids)==set(fitids)
        featurepath=FEATURES/(mode+'.npz');metapath=featurepath.with_suffix('.json')
        if not featurepath.exists():
            model=RecoveryModel(**saved['architecture']).eval();model.load_state_dict(saved['state'])
            xx=[];yy=[];pp=[]
            for index in np.r_[train,valid]:
                pid=ids[index];person=load_person(str(CACHE),pid)
                chosen=sample(len(person['length']),64)
                # Transform uses original metadata, then only chosen windows enter local encoder.
                b=saved['normalizer'].transform(person,'cpu','cnn')
                with torch.no_grad():z,present=model.local.encode(b['x'][chosen],b['length'][chosen])
                z=z.masked_fill(~present[:,:,None],0.).reshape(len(chosen),-1).numpy()
                with np.load(LOCAL/'physical_targets'/f'{pid}.npz') as data:
                    yy.append(data['window_targets'][chosen]);names=data['target_names']
                xx.append(z);pp.extend([pid]*len(chosen))
            np.savez_compressed(featurepath,x=np.concatenate(xx),y=np.concatenate(yy),participant_id=np.asarray(pp),target_names=names)
            write(metapath,dict(signature=sig['signature'],checkpoint_sha256=sha(checkpoint),features_sha256=sha(featurepath)))
            del model
        meta=json.loads(metapath.read_text());assert meta['signature']==sig['signature'] and meta['checkpoint_sha256']==sha(checkpoint)
        assert meta['features_sha256']==sha(featurepath)
        with np.load(featurepath) as data:x=data['x'];y=data['y'];people=data['participant_id'];names=data['target_names']
        finite=np.isfinite(y).all(1);coverage.append(dict(pool=mode,rows=len(x),defined_rows=int(finite.sum())))
        x=x[finite];y=y[finite];people=people[finite];assert not set(people)&set(ids[test])
        tr=np.flatnonzero(np.isin(people,fitids));te=np.flatnonzero(np.isin(people,valids))
        for method in CONFIG['probe_regressors']:
            output=DEST/f'{mode}_{method}.json'
            if output.exists():
                savedresult=json.loads(output.read_text());assert savedresult['signature']==sig['signature'];records.extend(savedresult['records']);continue
            scores=[]
            for alpha in CONFIG['probe_alphas']:
                for fold,(a,b) in enumerate(folds):
                    a=np.flatnonzero(np.isin(people,fitids[a]));b=np.flatnonzero(np.isin(people,fitids[b]))
                    prediction,_=fit_probe(x,y,people,a,b,alpha,method)
                    from sklearn.metrics import r2_score
                    scores.append(dict(alpha=alpha,fold=fold,r2=float(r2_score(y[b],prediction,sample_weight=weights(people[b])))))
            alpha=max(CONFIG['probe_alphas'],key=lambda a:np.mean([v['r2'] for v in scores if v['alpha']==a]))
            prediction,state=fit_probe(x,y,people,tr,te,alpha,method);w=weights(people[te]);rows=[]
            from sklearn.metrics import r2_score,mean_absolute_error
            for j,name in enumerate(names):
                rows.append(dict(pool=mode,method=method,target=name,dimensions=x.shape[1],alpha=alpha,
                    r2=float(r2_score(y[te,j],prediction[:,j],sample_weight=w)),mae=float(mean_absolute_error(y[te,j],prediction[:,j],sample_weight=w))))
            state.update(fit_ids=fitids.tolist(),validation_ids=valids.tolist(),signature=sig['signature'])
            path=FEATURES/f'{mode}_{method}.joblib';joblib.dump(state,path)
            transformed=state['x_scaler'].transform(x[te]);transformed=state['kernel'].transform(transformed) if state['kernel'] else transformed
            replay=state['y_scaler'].inverse_transform(state['model'].predict(transformed));error=float(np.max(np.abs(replay-prediction)));assert error<1e-10
            pd.DataFrame(dict(participant_id=people[te],**{f'true_{n}':y[te,j] for j,n in enumerate(names)},
                **{f'pred_{n}':prediction[:,j] for j,n in enumerate(names)})).to_csv(DEST/f'{mode}_{method}_predictions.csv',index=False)
            write(output,dict(signature=sig['signature'],records=rows,selection=scores,checkpoint=path.relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha(path),replay_error=error,fit_count=10));records.extend(rows)
            print('POOL PROBE',mode,method,np.mean([r['r2'] for r in rows]),flush=True)
    old=json.loads((OUT/'phase1_bottleneck/probes/validation.json').read_text())
    fits=100;assert fits+old['cpu_fits']<=RESOURCE['maximum_cpu_probe_fits']
    pd.DataFrame(records).to_csv(OUT/'tables/phase1_pooling_physical_probes.csv',index=False)
    pd.DataFrame(coverage).to_csv(DEST/'coverage.csv',index=False)
    write(DEST/'validation.json',dict(complete=True,signature=sig['signature'],cpu_fits=fits,
        total_phase1_cpu_probe_fits=fits+old['cpu_fits'],outer_test_loaded=False,
        train_target_scaling=True,inner_participant_selection=True,
        warning='Multi4 concatenated probe is128 dimensions; single-vector readouts32, capacity is not exactly matched'))


if __name__=='__main__':main()
