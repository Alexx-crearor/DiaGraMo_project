"""Auditable mappings and algebra for held-out model explanations.

Grouped sums are aggregated individual SHAP, not coalition/group Shapley values.
Fusion corrections are probability differences, not attention or causal effects.
"""
import numpy as np
import pandas as pd
from scipy.special import expit

GROUPS={'velocity':'Velocity','acceleration':'Acceleration','jerk':'Jerk','peaks':'Peaks',
        'lifts':'Lifts','pressure':'Pressure','orientation':'Orientation','trajectory':'Trajectory',
        'curvature':'Curvature','statistics':'Statistics','energy':'Energy','fft':'FFT'}


def feature_mapping(schema):
    tasks=schema['tasks']; features=schema['features_per_task']
    if len(tasks)!=14 or len(set(tasks))!=14 or len(features)!=74 or len(set(features))!=74:
        raise ValueError('Expected 14 unique tasks and 74 unique features')
    rows=[]
    for task_index,task in enumerate(tasks):
        for local_index,name in enumerate(features):
            prefix=name.split('_')[0]
            if prefix not in GROUPS: raise ValueError('Unknown source feature prefix: '+name)
            rows.append(dict(feature_index=len(rows),task_index=task_index,task=task,
                             feature_in_task=local_index,feature=name,group=GROUPS[prefix],
                             full_name=task+'::'+name))
    return pd.DataFrame(rows)


def align_shap(values,kept_indices,width=1036):
    values=np.asarray(values)
    indices=np.asarray(kept_indices)
    if values.ndim!=2 or values.shape[1]!=len(indices) or len(set(indices))!=len(indices):
        raise ValueError('SHAP/preprocessing width mismatch')
    if (indices<0).any() or (indices>=width).any(): raise ValueError('Invalid original feature indices')
    result=np.zeros((len(values),width),dtype=float)
    result[:,indices]=values
    return result


def validate_additivity(values,base,output,atol=2e-6):
    values=np.asarray(values); base=np.broadcast_to(np.asarray(base).reshape(-1), (len(values),))
    output=np.asarray(output).reshape(-1)
    if values.ndim!=2 or len(output)!=len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid attribution shape/values')
    error=float(np.max(np.abs(base+values.sum(axis=1)-output)))
    if error>atol: raise ValueError(f'Attribution additivity failed: {error}')
    return error


def aggregate_shap(values,mapping,level):
    """Keep signed sums and both absolute-mass and absolute-net definitions."""
    values=np.asarray(values)
    if values.shape[1]!=len(mapping): raise ValueError('Mapping width differs')
    names=list(dict.fromkeys(mapping[level]))
    signed=np.column_stack([values[:,mapping[level].eq(name)].sum(axis=1) for name in names])
    mass=np.column_stack([np.abs(values[:,mapping[level].eq(name)]).sum(axis=1) for name in names])
    np.testing.assert_allclose(signed.sum(axis=1),values.sum(axis=1),atol=1e-10)
    return names,signed,mass


def fusion_mechanism(pkin,pcwt,metadata,kind):
    """Exact two-branch probability algebra; never access labels."""
    pk=np.asarray(pkin,dtype=float); pc=np.asarray(pcwt,dtype=float)
    if pk.shape!=pc.shape or not np.isfinite(pk).all() or not np.isfinite(pc).all():
        raise ValueError('Invalid branch probabilities')
    if ((pk<0)|(pk>1)|(pc<0)|(pc>1)).any(): raise ValueError('Outside probability range')
    if kind=='late':
        weights=np.asarray(metadata['weights'])
        if len(weights)!=2 or (weights<0).any() or not np.isclose(weights.sum(),1):
            raise ValueError('Invalid late weights')
        wc=np.full(pk.shape,weights[1]); gate=np.full(pk.shape,False)
    elif kind=='adaptive':
        if len(metadata['weights'])!=1 or not np.isclose(metadata['weights'][0],1):
            raise ValueError('Expected one auxiliary branch')
        if not 0<=metadata['budget']<=.5: raise ValueError('Invalid correction budget')
        gate=np.abs(pk-.5)<=metadata['gate']
        wc=gate.astype(float)*metadata['budget']
    else: raise ValueError('Unsupported fusion mechanism')
    delta=wc*(pc-pk)
    return dict(probability=pk+delta,delta_cwt=delta,effective_weight_cwt=wc,
                effective_weight_kinematic=1-wc,gate_active=gate)


def outcomes(y,kinematic_prediction,fusion_prediction):
    y=np.asarray(y); k=np.asarray(kinematic_prediction)==y; f=np.asarray(fusion_prediction)==y
    return np.select([~k&f,k&~f,k&f],['corrected','harmed','unchanged_correct'],default='unchanged_wrong')


def linear_logit_contributions(model,Z,background):
    """Independent-reference linear decomposition; units are CWT log-odds.

No back-projection through pretrained encoders or claim of physiological meaning.
"""
    # Subtract in float64: float32 subtraction before multiplying float64
    # coefficients can break the exact linear identity by ~1e-7.
    Z=np.asarray(Z,dtype=np.float64); reference=np.asarray(background,dtype=np.float64).mean(axis=0)
    if model.coef_.shape!=(1,Z.shape[1]): raise ValueError('Expected binary linear classifier')
    phi=(Z-reference)*model.coef_[0]
    base=float(model.intercept_[0]+reference@model.coef_[0])
    validate_additivity(phi,base,model.decision_function(Z),atol=1e-8)
    np.testing.assert_allclose(expit(base+phi.sum(axis=1)),model.predict_proba(Z)[:,1],atol=1e-12)
    return phi,base


def permute_block(X,columns,permutation):
    """One joint row shuffle for the entire block; keep all other columns fixed."""
    X=np.asarray(X); columns=np.asarray(columns); permutation=np.asarray(permutation)
    if sorted(permutation.tolist())!=list(range(len(X))): raise ValueError('Not a permutation')
    result=X.copy()
    result[:,columns]=X[permutation][:,columns]
    return result


def exact_partition_explainers(model,background):
    """Lossless per-feature partition encoding for interventional TreeSHAP.

    SHAP 0.49's independent C++ engine stores split thresholds as float32.
    Binning at the *frozen model's* exact float64 cuts replaces thresholds by
    exactly representable half-integers, preserving every split for every input
    and every feature-masked coalition. No model fit or test-dependent cuts.
    The evaluated model/checkpoint is never modified. Leaf rounding remains
    subject to an explicit additivity check, not an offset correction.
    """
    import shap
    original=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw').model
    width=np.asarray(background).shape[1]; cuts=[[] for _ in range(width)]
    for tree in original.trees:
        for node in np.flatnonzero(tree.children_left>=0):
            if tree.threshold_types[node]!=0: raise ValueError('Only numeric tree splits supported')
            cuts[int(tree.features[node])].append(tree.thresholds[node])
    cuts=[np.unique(c).astype(np.float64) for c in cuts]
    trees=[]
    for tree in original.trees:
        item={k:np.array(getattr(tree,k),copy=True) for k in ('children_left','children_right','children_default',
              'features','thresholds','threshold_types','values','node_sample_weight')}
        for node in np.flatnonzero(tree.children_left>=0):
            feature=int(tree.features[node]); j=int(np.searchsorted(cuts[feature],tree.thresholds[node]))
            item['thresholds'][node]=j+.5
        trees.append(item)
    def encode(X):
        X=np.asarray(X,dtype=np.float64)
        if X.shape[1]!=width: raise ValueError('Partition encoding width mismatch')
        Z=np.empty_like(X)
        for j,c in enumerate(cuts):
            Z[:,j]=np.searchsorted(c,X[:,j],side='left')
            Z[np.isnan(X[:,j]),j]=np.nan
        return Z
    # Exhaustive structural boundary check, independent of evaluation values.
    boundaries=0
    for c in cuts:
        if len(c):
            if not np.isfinite(c).all(): raise ValueError('Nonfinite split threshold')
            for j,t in enumerate(c):
                x=np.array([np.nextafter(t,-np.inf),t,np.nextafter(t,np.inf)])
                np.testing.assert_array_equal(np.searchsorted(c,x,side='left')<=j+.5,x<=t)
                boundaries+=3
    representation=dict(trees=trees,internal_dtype=np.float64,input_dtype=np.float64,
                        objective=original.objective,tree_output=original.tree_output,base_offset=original.base_offset.copy())
    B=encode(background)
    raw=shap.TreeExplainer(representation,data=B,feature_perturbation='interventional',model_output='raw')
    probability=shap.TreeExplainer(representation,data=B,feature_perturbation='interventional',model_output='probability')
    return raw,probability,encode,dict(method='lossless frozen-tree partition bins',boundaries_checked=boundaries,
            total_unique_cuts=sum(map(len,cuts)),max_bins=max(map(len,cuts))+1,cuts=[c.tolist() for c in cuts])
