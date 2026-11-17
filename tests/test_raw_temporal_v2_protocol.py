"""Participant isolation, honest fusion inputs, and fixed label-count subsets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import fusion_raw_temporal_v2 as fusion
import raw_temporal_v2_evaluation as service
from configure_raw_temporal_v2_phase3 import labeled_subset


def test_label_subsets_nested_and_train_only():
    y=np.tile([0,1],30);train=np.arange(10,50)
    a=labeled_subset(train,y,.25,42);b=labeled_subset(train,y,.5,42)
    assert set(a)<=set(b)<=set(train)
    assert len(a)==10 and len(b)==20 and set(y[a])=={0,1}
    assert set(a)!=set(labeled_subset(train,y,.25,43))


def blocks(monkeypatch):
    ids=np.asarray(['P'+str(i) for i in range(6)])
    monkeypatch.setattr(fusion,'cohort',lambda:(ids,np.asarray([0,1]*3),{},None,None))
    fit=np.arange(4);validation=np.arange(4,6)
    kin=dict(fit_ids=ids[fit].tolist(),evaluation_ids=ids[validation].tolist(),experiment_id='kin')
    raw=dict(kin,experiment_id='raw',architecture={},fixed_epochs=12,validation_ids=[],normalizer_fit_ids=ids[fit].tolist())
    return ids,fit,validation,kin,raw


def test_crossfit_accepts_disjoint_fixed_recipe(monkeypatch):
    ids,fit,validation,kin,raw=blocks(monkeypatch)
    proof=fusion.assert_block(np.arange(6),fit,validation,kin,raw)
    assert not set(proof['fit_ids'])&set(proof['validation_ids'])


@pytest.mark.parametrize('failure',['early_stop','normalizer','overlap','fit_identity'])
def test_crossfit_rejects_leakage(monkeypatch,failure):
    ids,fit,validation,kin,raw=blocks(monkeypatch)
    if failure=='early_stop':raw.update(fixed_epochs=None,validation_ids=ids[validation].tolist())
    elif failure=='normalizer':raw['normalizer_fit_ids']=ids.tolist()
    elif failure=='overlap':validation=np.asarray([3,4])
    else:raw['fit_ids']=ids[[0,1,2,4]].tolist()
    with pytest.raises(AssertionError):fusion.assert_block(np.arange(6),fit,validation,kin,raw)


def test_weight_ties_retain_kinematic_anchor(monkeypatch):
    monkeypatch.setattr(service,'config',lambda:{'fusion':{'weight_grid':[0,.1,.5,1]}})
    y=np.asarray([0,0,1,1]);kin=np.asarray([.1,.4,.7,.9]);raw=1-kin
    assert service.weight_fit(y,np.column_stack([kin,raw]))==0


def test_probability_logit_clipping_finite():
    x=np.asarray([[0,.5],[1,.9]]);z=service.logit_features(x)
    assert np.isfinite(z).all() and z[0,1]==0
    np.testing.assert_allclose(1/(1+np.exp(-z)),np.clip(x,1e-6,1-1e-6),rtol=1e-10,atol=1e-12)


def test_fast_bootstrap_metrics_match_independent_library():
    from analyze_raw_temporal_v2_phase4 import fast_metrics
    from raw_temporal_v2_common import metrics
    y=np.asarray([0,0,1,1,0,1,0,1])
    for p in (np.full(8,.5),np.asarray([0,1,.5,.9,.1,.2,.7,1.])):
        a=fast_metrics(y,p);b=metrics(y,p)
        for name in b:assert abs(a[name]-b[name])<1e-12


def test_paired_error_categories_keep_participant_alignment():
    from analyze_raw_temporal_v2_phase4 import error_categories
    y=np.asarray([0,0,1,1]);kin=np.asarray([.1,.8,.8,.2]);raw=np.asarray([.7,.9,.7,.7])
    assert error_categories(y,kin,raw).tolist()==['kin_only','both_wrong','both_correct','raw_only']
