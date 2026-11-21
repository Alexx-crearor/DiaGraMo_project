"""Numerical audit helpers only; no changes to frozen training."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_locked_temporal_data import association
from report_locked_temporal_study import calibration,metrics


def test_calibration_probability_edges_and_empty_bins():
    table,ece=calibration(np.array([0,1,1]),np.array([0.,.5,1.]))
    assert table.n.sum()==3 and table.iloc[-1].n==1 and table.iloc[0].n==1
    assert table.loc[table.n==0,'observed_rate'].isna().all()
    assert np.isclose(ece,1/6)


def test_source_association_is_descriptive_and_handles_missing():
    data=pd.DataFrame(dict(source=['a','a','b',None],label=[0,0,1,1]))
    summary,table=association(data,'source')
    assert summary['participants']==4 and summary['descriptive_label_purity']==1.
    assert np.isclose(summary['cramers_v'],1.) and table.to_numpy().sum()==4
    assert '<missing>' in table.index


def test_existing_strict_threshold_is_preserved():
    scores=metrics(np.array([0,1]),np.array([.5,.51]))
    assert scores['accuracy']==1. and scores['f1_macro']==1.
