"""Leakage barriers, independent transforms and constrained fusion behavior."""
import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from diagramo.fusion import BranchBank, ProbabilityFusion, check_crossfit, choose_inner
from fusion_common import GRID, VARIANTS


class FusionTests(unittest.TestCase):
    def test_transform_reference_and_heldout_state(self):
        rng=np.random.default_rng(42)
        X=rng.normal(size=(70,90)); X[:,-1]=5; X[0,0]=np.nan
        y=np.tile([0,1],35); ids=np.array([str(i) for i in range(70)])
        bank=BranchBank().fit(X[:50],y[:50],ids[:50],['none','anova_64','pca_var95'])
        A=bank.pre_.transform(X[:50]); B=bank.pre_.transform(X[50:])
        actual=bank.transform_many(X[50:])
        np.testing.assert_array_equal(actual['anova_64'],SelectKBest(f_classif,k=64).fit(A,y[:50]).transform(B))
        np.testing.assert_allclose(actual['pca_var95'],PCA(n_components=.95,svd_solver='full').fit(A).transform(B),atol=1e-10)
        before=bank.pre_[2].mean_.copy(); pca=bank.pca_.components_.copy()
        bank.transform_many(np.full((20,90),1e10))
        np.testing.assert_array_equal(before,bank.pre_[2].mean_)
        np.testing.assert_array_equal(pca,bank.pca_.components_)
        self.assertEqual(bank.fit_ids_,ids[:50].tolist())
        with self.assertRaises(ValueError): BranchBank().fit(X,y,['same']*70,['none'])

    def test_crossfit_detects_overlap_and_coverage(self):
        ids=np.arange(30); y=np.tile([0,1],15)
        splits=list(StratifiedKFold(3,shuffle=True,random_state=42).split(ids,y))
        check_crossfit(ids,splits)
        with self.assertRaises(ValueError): check_crossfit(ids,splits[:2])
        with self.assertRaises(ValueError): check_crossfit(ids,[(np.arange(30),np.arange(30))])

    def test_crossfit_training_really_excludes_each_subject(self):
        # A held-out sentinel must not influence even an unsupervised scaler.
        X=np.arange(90,dtype=float).reshape(30,3); y=np.tile([0,1],15); ids=np.arange(30)
        splits=list(StratifiedKFold(3,shuffle=True,random_state=42).split(ids,y))
        for tr,va in splits:
            altered=X.copy(); altered[va]=1e20
            a=BranchBank().fit(X[tr],y[tr],ids[tr],['none'])
            b=BranchBank().fit(altered[tr],y[tr],ids[tr],['none'])
            np.testing.assert_array_equal(a.pre_[2].mean_,b.pre_[2].mean_)
            self.assertFalse(set(a.fit_ids_)&set(ids[va].astype(str)))

    def test_equal_convex_pool_and_learned_probability_bounds(self):
        rng=np.random.default_rng(9); P=rng.uniform(.01,.99,(60,3)); y=(P[:,0]>.5).astype(int)
        equal=ProbabilityFusion(kind='equal').fit(P,y)
        np.testing.assert_allclose(equal.predict_proba(P)[:,1],P.mean(axis=1))
        model=ProbabilityFusion(kind='late',ridge=0).fit(P,y)
        self.assertGreater(model.weights_[0],.99)
        self.assertAlmostEqual(model.weights_.sum(),1)
        self.assertTrue((model.weights_>=0).all())
        Q=model.predict_proba(P)[:,1]
        self.assertTrue((Q>=P.min(axis=1)-1e-10).all() and (Q<=P.max(axis=1)+1e-10).all())

    def test_adaptive_gate_budget_and_null(self):
        P=np.array([[.9,.1],[.55,.9],[.2,.9],[.45,.1]])
        y=np.array([1,1,0,0])
        m=ProbabilityFusion(kind='adaptive',budget=.25,gate=.1).fit(P,y)
        q=m.predict_proba(P)[:,1]
        np.testing.assert_array_equal(q[[0,2]],P[[0,2],0])
        self.assertTrue((np.abs(q-P[:,0])<=.25+1e-10).all())
        null=ProbabilityFusion(kind='adaptive',budget=0,gate=.5).fit(P,y)
        np.testing.assert_array_equal(null.predict_proba(P)[:,1],P[:,0])

    def test_selective_fallback_uses_inner_only(self):
        base=dict(candidate='K_only',mean_f1_macro=.8,mean_accuracy=.81,mean_auc=.9,mean_features=1036)
        row=dict(base,candidate='fusion',mean_f1_macro=.85,mean_accuracy=.80)
        self.assertEqual(choose_inner([row],base)['candidate'],'K_only')
        row['mean_accuracy']=.82
        self.assertEqual(choose_inner([row],base)['candidate'],'fusion')
        row['mean_f1_macro']=.79
        self.assertEqual(choose_inner([row],base)['candidate'],'K_only')

    def test_prespecified_grid_keeps_original_and_adapter_controls(self):
        for variant in VARIANTS:
            for method in ['compact','late','adaptive']:
                self.assertTrue(any(c['variant']==variant and c['method']==method for c in GRID.values()))
        for c in GRID.values():
            if c['method']=='original': self.assertNotIn('G',c['branches'])
            if c['method']=='adaptive': self.assertEqual(c['options'][0],'K__none__LightGBM')


if __name__=='__main__': unittest.main()
