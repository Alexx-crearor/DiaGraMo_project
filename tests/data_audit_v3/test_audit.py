"""Numerical, leakage-boundary and protected-artifact regression checks."""
import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/data_audit_v3'))
from common import ROOT,read,sha,metrics
from phase_abc import association,source_majority,category,label_table
from phase_d import quality,fdr
from phase_ef import calibration,wilson
from run import local_module


def test_majority_control_only_uses_train_and_handles_unseen():
    source=np.array(['a','a','b','c']);labels=np.array([0,0,1,1]);train=np.array([0,1,2]);test=np.array([3])
    before=source_majority(source[train],labels[train],source[test]);labels[test]=0
    after=source_majority(source[train],labels[train],source[test]);np.testing.assert_array_equal(before,after)
    assert np.isclose(before[0],2/5)
    actual=source_majority(source[train],labels[train],np.array(['a','b']))
    np.testing.assert_allclose(actual,[(0+2*.4)/4,(1+2*.4)/3])


def test_source_association_and_permutation_reproducible():
    labels=np.array([0,0,1,1]);values=pd.Series(['NA','NA','b','b'])
    first,_=association(values,labels,99);second,_=association(values,labels,99)
    assert first==second and np.isclose(first['cramers_v'],1.) and first['label_purity']==1.
    assert first['sparse_table'] and first['mixed_sources']==0
    independent,_=association(pd.Series(['a','b','a','b']),labels,99)
    assert np.isclose(independent['cramers_v'],0.)


def test_missing_diagnosis_is_not_invented_disagreement():
    data=pd.DataFrame(dict(participant_id=['a','b'],group=[0,1],diagnosis=[1,''],flag_to_omit_total=[1,0],
        id_city=['city1','city2'],id_school_ppp=['s1','s2'],site_type_ppp=[0,1],date_of_administration=['d','d'],
        example_flag_to_omit=[1,0]))
    labels=label_table(data)
    assert labels.label_disagreement.tolist()==[True,False]
    assert labels.diagnosis_missing.tolist()==[False,True]
    assert labels.observed_flag_components.tolist()==['example_flag_to_omit','']
    assert labels.component_or_matches_total.all()


def test_literal_na_prefix_and_missing_are_distinct():
    data=pd.read_csv(io.StringIO('source\nNA\ncity1\n""\n'),keep_default_na=False,skip_blank_lines=False)
    assert category(data.source).tolist()==['NA','city1','<missing>']


def test_single_class_metrics_are_not_fabricated_auc():
    result=metrics(np.array([1,1]),np.array([.8,.4]))
    assert np.isnan(result['auc']) and np.isnan(result['specificity']) and np.isnan(result['balanced_accuracy'])
    assert result['sensitivity']==.5 and result['n']==2
    assert metrics(np.array([0,1]),np.array([.5,.51]))['accuracy']==1.
    with pytest.raises(ValueError):metrics(np.array([0]),np.array([1.1]))


def test_calibration_empty_bins_and_binomial_support():
    table,ece=calibration(np.array([0,1,1]),np.array([0.,.5,1.]))
    assert table.n.sum()==3 and np.isclose(ece,1/6)
    assert table.loc[table.n==0,'observed_rate'].isna().all()
    low,high=wilson(7,7);assert low<.7 and high<=1.
    assert np.isnan(wilson(0,0)[0])


def test_sensor_statistics_preserve_invalid_time_and_constant_channels():
    a=np.zeros((4,7));a[:,2]=[0,.01,.01,2.02];a[:,3]=[1,1,0,0]
    result=quality(a)
    assert result['samples']==4 and result['duplicate_dt']==1 and result['large_gaps']==1
    assert np.isnan(result['pressure_min_observed_step'])
    assert np.isclose(result['pen_down_valid_duration'],.01)


def test_fdr_preserves_unknown_tests():
    result=fdr(np.array([.01,.04,np.nan,.2]))
    assert np.isnan(result[2]);np.testing.assert_allclose(result[[0,1,3]],[.03,.06,.2])


def test_protected_baseline_source_and_oof_regression():
    baseline=read('PROTECTED_BEFORE.csv').set_index('path')
    for name in ['scripts/prepare.py','src/diagramo/features.py','src/diagramo/temporal/preprocessing.py',
        'src/diagramo/temporal/dataset.py','results/cv/fold_assignments.csv','results/cv/kinematic__LightGBM.oof.csv',
        'results/raw_temporal_v2/locked_study/participant_complementarity.csv','data/derived/kinematic.npz']:
        assert sha(ROOT/name)==baseline.loc[name,'sha256']


def test_cli_loads_only_audit_modules_not_original_reports():
    for name in ['report','validate']:
        module=local_module(name)
        assert Path(module.__file__).resolve().parent==ROOT/'scripts/data_audit_v3'
    with pytest.raises(ValueError):local_module('../report')
