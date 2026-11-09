"""Output-space correctness, feature/task alignment and exact fusion mechanics."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import shap

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from diagramo.explain import feature_mapping,align_shap,aggregate_shap,validate_additivity,permute_block,fusion_mechanism,outcomes,linear_logit_contributions,exact_partition_explainers
from diagramo.fusion import ProbabilityFusion


class ExplainTests(unittest.TestCase):
    def test_partition_encoding_preserves_threshold_boundary_and_shap_game(self):
        t=1.00000003
        tree=dict(children_left=np.array([1,-1,-1]),children_right=np.array([2,-1,-1]),
            children_default=np.array([1,-1,-1]),features=np.array([0,-1,-1]),thresholds=np.array([t,-1,-1]),
            threshold_types=np.array([0,-1,-1]),values=np.array([[0.],[-2.],[3.]]),node_sample_weight=np.array([4.,2.,2.]))
        model=dict(trees=[tree],objective='binary_crossentropy',tree_output='log_odds',base_offset=0.)
        background=np.array([[t-1e-8,0.],[t+1e-8,1.]])
        X=np.array([[np.nextafter(t,-np.inf),4.],[t,8.],[np.nextafter(t,np.inf),9.]])
        raw,prob,encode,audit=exact_partition_explainers(model,background)
        target=np.where(X[:,0]<=t,-2.,3.)
        np.testing.assert_array_equal(raw.model.predict(encode(X)),target)
        r=raw(encode(X)); p=prob(encode(X))
        validate_additivity(r.values,r.base_values,target,atol=1e-12)
        validate_additivity(p.values,p.base_values,expit(target),atol=1e-12)
        np.testing.assert_allclose(r.values[:,0],target-.5,atol=1e-12)
        np.testing.assert_array_equal(r.values[:,1],0)
        self.assertEqual(tree['thresholds'][0],t)
        self.assertEqual(audit['boundaries_checked'],3)

    def test_mapping_uses_schema_order_and_partitions_all_columns(self):
        schema=json.loads((ROOT/'data/derived/schema.json').read_text())
        schema['tasks']=list(reversed(schema['tasks']))
        m=feature_mapping(schema)
        self.assertEqual(len(m),1036); self.assertEqual(m.full_name.nunique(),1036)
        self.assertEqual(m.iloc[0].task,schema['tasks'][0]); self.assertEqual(m.iloc[74].feature_index,74)
        self.assertTrue((m.groupby('task').size()==74).all())
        self.assertEqual(m.group.nunique(),12)
        self.assertTrue(m[m.feature.str.startswith('fft_')].group.eq('FFT').all())
        with self.assertRaises(ValueError): feature_mapping(dict(schema,tasks=['copy']*14))

    def test_removed_constant_alignment_and_additive_grouping(self):
        m=feature_mapping(json.loads((ROOT/'data/derived/schema.json').read_text()))
        values=align_shap(np.array([[2.,-2.,3.]]),[0,1,74])
        self.assertEqual(values.shape,(1,1036)); self.assertEqual(values[0,2],0)
        names,signed,mass=aggregate_shap(values,m,'task')
        self.assertEqual(signed[0,0],0); self.assertEqual(mass[0,0],4)
        self.assertEqual(signed.sum(),3); self.assertEqual(mass.sum(),7)
        with self.assertRaises(ValueError): align_shap(np.ones((2,2)),[0,0])

    def test_tree_shap_raw_and_probability_units(self):
        rng=np.random.default_rng(42); X=rng.normal(size=(90,5)); y=(X[:,0]+.5*X[:,1]>.1).astype(int)
        model=LGBMClassifier(n_estimators=25,num_leaves=5,min_child_samples=4,verbosity=-1,n_jobs=1,random_state=42).fit(X[:70],y[:70])
        bg=X[:50]; test=X[70:]
        raw=shap.TreeExplainer(model,data=bg,feature_perturbation='interventional',model_output='raw')(test)
        prob=shap.TreeExplainer(model,data=bg,feature_perturbation='interventional',model_output='probability')(test)
        margin=model.predict(test,raw_score=True)
        validate_additivity(raw.values,raw.base_values,margin)
        validate_additivity(prob.values,prob.base_values,model.predict_proba(test)[:,1])
        np.testing.assert_allclose(expit(margin),model.predict_proba(test)[:,1],atol=1e-12)
        self.assertFalse(np.allclose(expit(raw.values),prob.values))
        with self.assertRaises(ValueError): validate_additivity(raw.values,raw.base_values,model.predict_proba(test)[:,1])

    def test_joint_permutation_preserves_block_and_other_features(self):
        X=np.arange(30).reshape(6,5); permutation=np.array([5,0,4,1,3,2])
        changed=permute_block(X,[1,3],permutation)
        np.testing.assert_array_equal(changed[:,[1,3]],X[permutation][:,[1,3]])
        np.testing.assert_array_equal(changed[:,[0,2,4]],X[:,[0,2,4]])
        np.testing.assert_array_equal(X,np.arange(30).reshape(6,5))
        with self.assertRaises(ValueError): permute_block(X,[1],[0]*6)

    def test_adaptive_is_gated_correction_not_attention(self):
        pk=np.array([.9,.6,.5,.4,.1]); pc=np.array([.1,.1,.9,.9,.9]); y=np.array([1,0,1,1,0])
        model=ProbabilityFusion(kind='adaptive',gate=.1,budget=.25).fit(np.column_stack([pk,pc]),y)
        r=fusion_mechanism(pk,pc,model.metadata(),'adaptive')
        np.testing.assert_allclose(r['probability'],model.predict_proba(np.column_stack([pk,pc]))[:,1])
        np.testing.assert_allclose(r['delta_cwt'],r['effective_weight_cwt']*(pc-pk))
        self.assertEqual(r['effective_weight_cwt'][0],0); self.assertEqual(r['delta_cwt'][0],0)
        self.assertAlmostEqual(r['effective_weight_cwt'][1],.25)

    def test_late_decomposition_and_outcomes(self):
        pk=np.array([.8,.8,.2,.2]); pc=np.array([.1,.1,.9,.9])
        r=fusion_mechanism(pk,pc,dict(weights=[.7,.3]),'late')
        np.testing.assert_allclose(r['probability'],.7*pk+.3*pc)
        np.testing.assert_allclose(r['probability']-pk,r['delta_cwt'])
        status=outcomes([1,0,1,0],[0,0,1,1],[1,1,1,1])
        self.assertEqual(status.tolist(),['corrected','harmed','unchanged_correct','unchanged_wrong'])

    def test_cwt_linear_contribution_is_logit_not_signal_attribution(self):
        rng=np.random.default_rng(42); Z=rng.normal(size=(70,6)); y=(Z[:,0]+Z[:,2]>0).astype(int)
        model=LogisticRegression().fit(Z[:50],y[:50]); phi,base=linear_logit_contributions(model,Z[50:],Z[:30])
        validate_additivity(phi,base,model.decision_function(Z[50:]),atol=1e-10)
        independent=shap.LinearExplainer(model,shap.maskers.Independent(Z[:30]))(Z[50:])
        np.testing.assert_allclose(phi,independent.values,atol=1e-12)
        np.testing.assert_allclose(base,independent.base_values,atol=1e-12)
        # Cached pretrained embeddings are float32; their subtraction must not
        # silently lose the linear identity before multiplication by coefficients.
        p32,b32=linear_logit_contributions(model,Z[50:].astype('float32'),Z[:30].astype('float32'))
        validate_additivity(p32,b32,model.decision_function(Z[50:].astype('float32')),atol=1e-10)


if __name__=='__main__': unittest.main()
