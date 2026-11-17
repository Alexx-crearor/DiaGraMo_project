"""Frozen v1 layer probes, participant-disjoint development validation.

Physical targets describe each observation's own scope. Rows without defined
motion targets are excluded jointly, never replaced by artificial zero motion.
CPU execution can accompany the single GPU training worker.
"""
import argparse
import time
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from raw_temporal_v2_common import *
from diagramo.temporal.dataset import load_person,collate_people
from diagramo.temporal.hierarchical_model import RawTemporalModel
from diagramo.temporal_v2.models import RecoveryModel

CONFIG_PATH='configs/raw_temporal_v2/phase1.json'
CONFIG=json.loads((ROOT/CONFIG_PATH).read_text())
DEST=OUT/'phase1_bottleneck/probes'
FEATURES=LOCAL/'probe_features'
SCRIPT='scripts/probe_raw_temporal_v2.py'


def sample(count,cap):return np.unique(np.linspace(0,count-1,min(count,cap)).astype(int))


def extract():
    seed_all(42);torch.set_num_threads(2)
    ids,y,schema,fit,val,test=development();allowed=np.r_[fit,val]
    sig=signature(CONFIG_PATH,[SCRIPT,'src/diagramo/temporal_v2/models.py'])
    FEATURES.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    manifest={};checks=[]
    for family,name in [('cnn','cnn_1_0'),('hierarchical','hierarchical_0_0')]:
        path=OLD/'checkpoints/outer_0/inner'/f'{name}.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert set(state['normalizer'].fit_ids)==set(ids[fit])
        original=RawTemporalModel(**state['architecture']).eval();original.load_state_dict(state['state'])
        model=RecoveryModel(**state['architecture']).eval()
        changed={('local.backbone.'+key[6:] if key.startswith('local.') else key):value for key,value in state['state'].items()}
        model.base.load_state_dict(changed)
        expected=pd.read_csv(OLD/'folds/fold_0/inner'/f'{name}.oof.csv').set_index('participant_id')
        pools=defaultdict(list);targets=defaultdict(list);participants=defaultdict(list);reference={}
        errors=[]
        for ii,index in enumerate(allowed):
            pid=ids[index];item=state['normalizer'].transform(load_person(str(CACHE),pid),'cpu',family)
            batch=collate_people([item])
            with torch.no_grad():
                logit,details=model(batch,return_details=True)
                values,valid=model.local.feature_map(batch['x'],batch['length'])
                baseline=original(batch)
            assert float((baseline-logit).abs().max())<1e-5
            if index in val:
                p=float(logit.sigmoid());errors.append(abs(p-float(expected.loc[pid,'probability'])))
            with np.load(LOCAL/'physical_targets'/f'{pid}.npz') as physical:
                names=physical['target_names'].tolist()
                levels={'feature_map':('window',values.transpose(1,2).reshape(len(values),-1).numpy()),
                    'window':('window',details['window_tokens'][:,0].numpy())}
                if family=='hierarchical':levels.update(segment=('segment',details['segment'].numpy()),
                    task=('task',details['task'][0].numpy()),participant=('person',details['person'].numpy()))
                for level,(scope,x) in levels.items():
                    cap=CONFIG['probe_max_windows_per_participant'] if scope=='window' else CONFIG['probe_max_segments_per_participant'] if scope=='segment' else len(x)
                    chosen=sample(len(x),cap);target=physical[scope+'_targets'][chosen]
                    key=family+'_'+level
                    pools[key].append(x[chosen]);targets[key].append(target);participants[key].extend([pid]*len(chosen))
                    if scope not in reference:
                        reference[scope]=True
                    # Shared reference saved once per family/scope, not once per level.
                for scope in sorted(reference):
                    x=physical[scope+'_reference'];cap=64 if scope in ('window','segment') else len(x);chosen=sample(len(x),cap)
                    key=family+'_reference_'+scope
                    pools[key].append(x[chosen]);targets[key].append(physical[scope+'_targets'][chosen]);participants[key].extend([pid]*len(chosen))
            if ii%25==0:print('PROBE EXTRACT',family,ii+1,flush=True)
        assert max(errors)<1e-5,'Original held-out prediction replay failed'
        checks.append(dict(family=family,checkpoint=path.relative_to(ROOT).as_posix(),sha256=sha(path),
            maximum_validation_probability_replay_error=max(errors),fit_ids=ids[fit].tolist(),validation_ids=ids[val].tolist()))
        for key in pools:
            path=FEATURES/(key+'.npz')
            np.savez_compressed(path,x=np.concatenate(pools[key]),y=np.concatenate(targets[key]),
                participant_id=np.asarray(participants[key]),target_names=np.asarray(names))
            manifest[key]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),rows=len(participants[key]))
    write(DEST/'feature_manifest.json',dict(signature=sig,features=manifest,checkpoint_replay=checks,
        outer_test_loaded=False,complete=True,warning='Development validation also used for historical encoder early stopping; exploratory probes'))


def weights(ids):
    unique,count=np.unique(ids,return_counts=True);lookup=dict(zip(unique,count))
    result=np.asarray([1/lookup[i] for i in ids]);return result/result.mean()


def fit_probe(x,y,ids,train,test,alpha,method):
    w=weights(ids[train]);sx=StandardScaler().fit(x[train],sample_weight=w)
    sy=StandardScaler().fit(y[train],sample_weight=w)
    a=sx.transform(x[train]);b=sx.transform(x[test]);kernel=None
    if method=='rbf_ridge':
        kernel=RBFSampler(gamma=1/x.shape[1],n_components=CONFIG['probe_nonlinear_components'],random_state=42)
        a=kernel.fit_transform(a);b=kernel.transform(b)
    model=Ridge(alpha=alpha,solver='lsqr',tol=1e-5).fit(a,sy.transform(y[train]),sample_weight=w)
    return sy.inverse_transform(model.predict(b)),dict(x_scaler=sx,y_scaler=sy,kernel=kernel,model=model)


def run():
    import joblib
    seed_all(42);torch.set_num_threads(2);DEST.mkdir(parents=True,exist_ok=True)
    if not (DEST/'feature_manifest.json').exists():extract()
    manifest=json.loads((DEST/'feature_manifest.json').read_text())
    sig=signature(CONFIG_PATH,[SCRIPT,'src/diagramo/temporal_v2/models.py'])
    assert manifest['signature']['signature']==sig['signature']
    ids,labels,schema,fit,val,test=development();fitids=ids[fit];valids=ids[val]
    folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(fitids,labels[fit]))
    completed=DEST/'validation.json'
    if completed.exists():assert json.loads(completed.read_text())['signature']==sig['signature'];print('REUSE PROBES');return
    records=[];coverage=[];selection=[];fit_count=0;start=time.perf_counter()
    for level,artifact in manifest['features'].items():
        assert sha(ROOT/artifact['path'])==artifact['sha256']
        with np.load(ROOT/artifact['path']) as data:
            x=data['x'];y=data['y'];people=data['participant_id'];names=data['target_names']
        finite=np.isfinite(y).all(1)&np.isfinite(x).all(1)
        coverage.append(dict(level=level,rows=len(x),defined_rows=int(finite.sum()),
            excluded_undefined_motion_rows=int((~finite).sum()),participants=len(np.unique(people[finite]))))
        x=x[finite];y=y[finite];people=people[finite]
        train=np.flatnonzero(np.isin(people,fitids));testrows=np.flatnonzero(np.isin(people,valids))
        assert not set(people)&set(ids[test])
        for method in CONFIG['probe_regressors']:
            path=DEST/f'{level}_{method}.json'
            if path.exists():
                saved=json.loads(path.read_text());assert saved['signature']==sig['signature']
                records.extend(saved['records']);selection.extend(saved['selection']);fit_count+=saved['fits'];continue
            scores=[];fits=0
            for alpha in CONFIG['probe_alphas']:
                for fold,(a,b) in enumerate(folds):
                    tr=np.flatnonzero(np.isin(people,fitids[a]));va=np.flatnonzero(np.isin(people,fitids[b]))
                    prediction,_=fit_probe(x,y,people,tr,va,alpha,method);fits+=1
                    score=float(r2_score(y[va],prediction,sample_weight=weights(people[va]),multioutput='uniform_average'))
                    scores.append(dict(level=level,method=method,alpha=alpha,inner_fold=fold,r2=score))
            best=max(CONFIG['probe_alphas'],key=lambda a:np.mean([r['r2'] for r in scores if r['alpha']==a]))
            prediction,state=fit_probe(x,y,people,train,testrows,best,method);fits+=1
            assert fit_count+fits<=RESOURCE['maximum_cpu_probe_fits']
            w=weights(people[testrows]);rows=[]
            for j,name in enumerate(names):
                truth=y[testrows,j];pred=prediction[:,j]
                am=np.average(truth,weights=w);bm=np.average(pred,weights=w)
                covariance=np.average((truth-am)*(pred-bm),weights=w)
                denominator=np.sqrt(np.average((truth-am)**2,weights=w)*np.average((pred-bm)**2,weights=w))
                rows.append(dict(level=level,method=method,target=name,alpha=best,
                    r2=float(r2_score(truth,pred,sample_weight=w)),mae=float(mean_absolute_error(truth,pred,sample_weight=w)),
                    correlation=float(covariance/denominator) if denominator>0 else 0.,dimensions=x.shape[1],
                    fit_participants=len(np.unique(people[train])),validation_participants=len(np.unique(people[testrows])),
                    observation_scope='Within-observation target; participant-balanced row weights'))
            pd.DataFrame(dict(participant_id=people[testrows],**{f'true_{n}':y[testrows,j] for j,n in enumerate(names)},
                **{f'pred_{n}':prediction[:,j] for j,n in enumerate(names)})).to_csv(DEST/f'{level}_{method}_predictions.csv',index=False)
            state.update(fit_ids=fitids.tolist(),validation_ids=valids.tolist(),signature=sig['signature'])
            ckpt=FEATURES/f'{level}_{method}.joblib';joblib.dump(state,ckpt)
            write(path,dict(signature=sig['signature'],records=rows,selection=scores,fits=fits,
                checkpoint=ckpt.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(ckpt)))
            records.extend(rows);selection.extend(scores);fit_count+=fits
            print('PROBE FIT',level,method,'meanR2',round(np.mean([r['r2'] for r in rows]),4),flush=True)
    pd.DataFrame(records).to_csv(OUT/'tables/phase1_layer_probes.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'tables/phase1_probe_coverage.csv',index=False)
    pd.DataFrame(selection).to_csv(DEST/'inner_selection.csv',index=False)
    write(completed,dict(complete=True,signature=sig['signature'],cpu_fits=fit_count,
        seconds=time.perf_counter()-start,outer_test_loaded=False,targets_standardized_train_only=True,
        inner_selection_participant_disjoint=True,participant_balanced=True,
        undefined_targets='Complete-case physical-motion observations; coverage reported, no zero imputation',
        results_not_independent_confirmation=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract-only',action='store_true');args=parser.parse_args()
    if args.extract_only:extract()
    else:run()
