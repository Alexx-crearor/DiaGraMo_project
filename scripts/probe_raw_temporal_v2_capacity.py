"""Train-only PCA32 control for feature-map versus mean-window probe capacity."""
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from threadpoolctl import threadpool_limits
from probe_raw_temporal_v2 import weights
from raw_temporal_v2_common import *

DEST=OUT/'phase1_bottleneck/capacity_probes'


def regression(x,y,people,train,test,alpha,method):
    w=weights(people[train]);sx=StandardScaler().fit(x[train],sample_weight=w);sy=StandardScaler().fit(y[train],sample_weight=w)
    a=sx.transform(x[train]);b=sx.transform(x[test]);pca=PCA(32,svd_solver='randomized',random_state=42).fit(a)
    a=pca.transform(a);b=pca.transform(b);kernel=None
    if method=='rbf_ridge':
        kernel=RBFSampler(gamma=1/32,n_components=128,random_state=42).fit(a);a=kernel.transform(a);b=kernel.transform(b)
    model=Ridge(alpha=alpha,solver='lsqr',tol=1e-5).fit(a,sy.transform(y[train]),sample_weight=w)
    return sy.inverse_transform(model.predict(b)),dict(x_scaler=sx,y_scaler=sy,pca=pca,kernel=kernel,model=model)


def main():
    seed_all(42);DEST.mkdir(parents=True,exist_ok=True);ids,labels,schema,fit,val,test=development();records=[]
    sig=signature('configs/raw_temporal_v2/phase1.json',['scripts/probe_raw_temporal_v2_capacity.py','scripts/probe_raw_temporal_v2.py'])
    manifest=json.loads((OUT/'phase1_bottleneck/probes/feature_manifest.json').read_text())
    for family in ('cnn','hierarchical'):
        info=manifest['features'][family+'_feature_map'];assert sha(ROOT/info['path'])==info['sha256']
        with np.load(ROOT/info['path']) as data:x=data['x'];y=data['y'];people=data['participant_id'];names=data['target_names']
        finite=np.isfinite(y).all(1);x=x[finite];y=y[finite];people=people[finite]
        train=np.flatnonzero(np.isin(people,ids[fit]));testrows=np.flatnonzero(np.isin(people,ids[val]));assert not set(people)&set(ids[test])
        folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(ids[fit],labels[fit]))
        for method in ('ridge','rbf_ridge'):
            target=DEST/f'{family}_{method}.json'
            if target.exists():
                saved=json.loads(target.read_text());assert saved['signature']==sig['signature'];records.extend(saved['records']);continue
            scores=[]
            for alpha in (.1,10.,1000.):
                for fold,(a,b) in enumerate(folds):
                    a=np.flatnonzero(np.isin(people,ids[fit[a]]));b=np.flatnonzero(np.isin(people,ids[fit[b]]))
                    prediction,_=regression(x,y,people,a,b,alpha,method)
                    scores.append(dict(alpha=alpha,fold=fold,r2=float(r2_score(y[b],prediction,sample_weight=weights(people[b])))))
            alpha=max((.1,10.,1000.),key=lambda a:np.mean([r['r2'] for r in scores if r['alpha']==a]))
            prediction,state=regression(x,y,people,train,testrows,alpha,method);w=weights(people[testrows]);rows=[]
            for j,name in enumerate(names):
                rows.append(dict(level=family+'_feature_map_pca32',method=method,target=name,dimensions=32,alpha=alpha,
                    r2=float(r2_score(y[testrows,j],prediction[:,j],sample_weight=w)),
                    mae=float(mean_absolute_error(y[testrows,j],prediction[:,j],sample_weight=w))))
            state.update(fit_ids=ids[fit].tolist(),validation_ids=ids[val].tolist(),signature=sig['signature'])
            path=LOCAL/'probe_features'/f'{family}_pca32_{method}.joblib';joblib.dump(state,path)
            pd.DataFrame(dict(participant_id=people[testrows],**{f'true_{n}':y[testrows,j] for j,n in enumerate(names)},
                **{f'pred_{n}':prediction[:,j] for j,n in enumerate(names)})).to_csv(DEST/f'{family}_{method}_predictions.csv',index=False)
            write(target,dict(signature=sig['signature'],records=rows,selection=scores,checkpoint=path.relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha(path),pca_train_only=True,explained_variance_ratio_sum=float(state['pca'].explained_variance_ratio_.sum())))
            records.extend(rows);print('PCA32 PROBE',family,method,np.mean([r['r2'] for r in rows]),flush=True)
    pd.DataFrame(records).to_csv(OUT/'tables/phase1_capacity_probes.csv',index=False)
    previous=json.loads((OUT/'phase1_bottleneck/pooling_probes/validation.json').read_text())
    total=previous['total_phase1_cpu_probe_fits']+40;assert total<=RESOURCE['maximum_cpu_probe_fits']
    write(DEST/'validation.json',dict(complete=True,signature=sig['signature'],cpu_fits=40,total_phase1_cpu_probe_fits=total,
        pca_and_scalers_fit_train_only=True,outer_test_loaded=False,
        limitation='PCA is unsupervised and row-weighted after participant-balanced scaling; capacity matched32 dims, not learned projection equivalence'))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
