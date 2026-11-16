"""Meaningful timestamp, hierarchy, masking, batching and Train-only checks."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from diagramo.temporal.preprocessing import preprocess,CHANNELS
from diagramo.temporal.segmentation import windows,padded,groups
from diagramo.temporal.dataset import TrainNormalizer,collate_people
from diagramo.temporal.hierarchical_model import RawTemporalModel
torch.set_num_threads(2)

def synthetic():
    P=42;L=16;C=len(CHANNELS);task=np.repeat(np.arange(14),3);segment=np.arange(P)
    sample=dict(x=torch.randn(P,C,L),length=torch.full((P,),L),patch_task=torch.tensor(task),patch_segment=torch.tensor(segment),
      segment_task=torch.tensor(task),patch_meta=torch.randn(P,4),segment_meta=torch.randn(P,4),
      stroke_groups=torch.tensor(groups(segment,16)),stroke_group_segment=torch.arange(P),
      task_groups=torch.tensor(groups(task,32)),task_group_task=torch.arange(14),
      flat_groups=torch.tensor(groups(task,32)),flat_group_task=torch.arange(14))
    return sample

class RawTemporalTests(unittest.TestCase):
    def record(self):
        return dict(x=[0,1,2,4,5,10],y=[0,2,2,4,5,10],time=[0,.01,.01,.02,.03,2.],pen_status=[1,1,1,0,1,1],pressure=[2]*6,tilt=[40]*6,azimuth=[180]*6)
    def test_timestamp_policy_keeps_duplicates_and_guards_derivatives(self):
        r=self.record();p=preprocess(r)
        np.testing.assert_array_equal(p['time'],r['time']);np.testing.assert_array_equal(p['raw_index'],np.arange(6))
        self.assertEqual(p['physical'][1,8],100.)
        self.assertTrue((p['physical'][[0,2,3,4,5],8:13]==0).all())
        self.assertFalse(p['velocity_valid'][2]);self.assertTrue(p['boundary'][2]);self.assertTrue(p['boundary'][5])
    def test_negative_and_tiny_intervals_are_explicit_boundaries(self):
        r=self.record();r['time']=[0,.01,.009,.00900001,.01,.02];p=preprocess(r)
        self.assertTrue(p['boundary'][2]and p['boundary'][3]);self.assertEqual(p['physical'][2,17],1);self.assertEqual(p['physical'][3,18],1)
        self.assertTrue(np.isfinite(p['values']).all())
    def test_segmentation_inventory_has_exact_coverage_and_padding(self):
        p=preprocess(self.record());inv=windows(p,2);x,ix,n=padded(p,inv,2)
        indices=np.concatenate([np.arange(int(a),int(b))for a,b,*_ in inv]);np.testing.assert_array_equal(indices,np.arange(6))
        for i,(a,b,*_)in enumerate(inv):np.testing.assert_array_equal(x[i,:n[i]],p['values'][int(a):int(b)]);self.assertTrue((x[i,n[i]:]==0).all())
        self.assertEqual(int(n.sum()),6)
    def test_group_chunks_never_mix_strokes_or_drop_tokens(self):
        idx=np.r_[np.zeros(40),np.ones(3),np.full(22,2)];g=groups(idx,16)
        np.testing.assert_array_equal(g[g>=0],np.arange(65))
        for row in g:self.assertEqual(len(np.unique(idx[row[row>=0]])),1)
    def test_all_model_families_batch_matches_individual(self):
        a=synthetic();b=synthetic();batch=collate_people([a,b])
        for family in ('cnn','tcn','transformer','hierarchical','no_explicit_time','no_position','index_time','local_stroke'):
            torch.manual_seed(42);m=RawTemporalModel(family).eval()
            with torch.no_grad():single=torch.cat([m(a),m(b)]);combined=m(batch)
            torch.testing.assert_close(single,combined,atol=2e-6,rtol=0)
            with torch.no_grad():eight=m(collate_people([a,b]*4))
            torch.testing.assert_close(eight,combined.repeat(4),atol=3e-6,rtol=0)
    def test_padding_cannot_affect_local_representation(self):
        a=synthetic();a['length'][:]=9;b={k:v.clone()for k,v in a.items()};b['x'][:,:,9:]=1e5
        m=RawTemporalModel().eval()
        with torch.no_grad():torch.testing.assert_close(m(a),m(b),atol=1e-6,rtol=0)
    def test_missing_patch_input_noise_has_no_effect(self):
        a=synthetic();b={k:v.clone()for k,v in a.items()};mask=torch.ones(42,dtype=torch.bool);mask[3:6]=False;b['x'][3:6]=1e5
        m=RawTemporalModel().eval()
        with torch.no_grad():torch.testing.assert_close(m(a,patch_present=mask),m(b,patch_present=mask),atol=2e-6,rtol=0)
    def test_cached_segment_occlusion_matches_full_masked_forward(self):
        a=synthetic();m=RawTemporalModel().eval();mask=torch.ones(42,dtype=torch.bool);mask[[2,9]]=False
        with torch.no_grad():
            seg,present=m(a,return_segments=True);p=m.classify_segments(a,seg,mask);ref=m(a,patch_present=mask)
            torch.testing.assert_close(p,ref,atol=2e-6,rtol=0)
            many=torch.stack([present,mask]);batch=m.classify_segments(a,seg,many);torch.testing.assert_close(batch,torch.cat([m(a),ref]),atol=2e-6,rtol=0)
    def test_real_time_encoding_disabled_control(self):
        a=synthetic();b={k:v.clone()for k,v in a.items()};b['patch_meta']+=2;b['segment_meta']+=2
        torch.manual_seed(4);m=RawTemporalModel('no_position').eval()
        with torch.no_grad():torch.testing.assert_close(m(a),m(b),atol=1e-7,rtol=0)
        torch.manual_seed(4);real=RawTemporalModel('hierarchical').eval()
        with torch.no_grad():self.assertGreater(float(abs(real(a)-real(b))),1e-6)
    def test_train_normalization_ignores_forbidden_subject(self):
        stats=np.zeros((3,4,21));stats[:,1]=1.;stats[:,3]=1.;stats[2]=1e12
        norm=TrainNormalizer().fit(stats,[0,1],['A','B']);self.assertEqual(norm.fit_ids,['A','B']);self.assertLess(norm.mean.max(),1.)
        self.assertGreaterEqual(norm.scale.min(),.1)
    def test_models_are_small_and_have_no_chronological_task_encoding(self):
        m=RawTemporalModel().eval();self.assertLess(sum(p.numel()for p in m.parameters()),50000)
        self.assertEqual(m.task_id.num_embeddings,14);self.assertFalse(hasattr(m,'task_position'))
        # Shuffle already identity-tagged tasks, keeping their masks attached.
        # Chronological task positions would break this actual aggregation test.
        task=m.task_ff(torch.randn(1,14,32)+m.task_id(torch.arange(14))[None])
        valid=torch.ones(1,14,dtype=torch.bool);valid[0,:2]=False;order=torch.randperm(14)
        def aggregate(tokens,mask):
            contextual,_=m.person_context(tokens,mask);person,_=m.pool(contextual,mask)
            return m.head(person)
        with torch.no_grad():torch.testing.assert_close(aggregate(task,valid),aggregate(task[:,order],valid[:,order]),atol=1e-6,rtol=0)

    def test_joint_ig_linear_inputs_and_time_metadata_are_additive(self):
        from diagramo.temporal.explain import integrated_gradients
        class Linear(torch.nn.Module):
            def forward(self,b):return .1*b['x'].sum()+.2*b['patch_meta'].sum()-.3*b['segment_meta'].sum()+.5
        b={'x':torch.rand(2,3,4),'patch_meta':torch.rand(2,4),'segment_meta':torch.rand(2,4)}
        r=integrated_gradients(Linear(),b,steps=8)
        np.testing.assert_allclose(r['x'],b['x'].numpy()*.1,atol=1e-7)
        np.testing.assert_allclose(r['patch_meta'],b['patch_meta'].numpy()*.2,atol=1e-7)
        np.testing.assert_allclose(r['segment_meta'],b['segment_meta'].numpy()*-.3,atol=1e-7)
        self.assertLess(abs(r['residual']),2e-6)

    def test_structure_reference_keeps_pen_and_validity_signals(self):
        from diagramo.temporal.explain import references
        b=synthetic();ref=references(b,'structure_preserving')
        torch.testing.assert_close(ref['x'][:,7],b['x'][:,7]);torch.testing.assert_close(ref['x'][:,13:],b['x'][:,13:])
        self.assertEqual(float(ref['x'][:,:7].abs().sum()),0.)
        torch.testing.assert_close(ref['patch_meta'][:,3],b['patch_meta'][:,3])

    def test_actual_hierarchy_ig_completeness(self):
        from diagramo.temporal.explain import converged_ig
        torch.manual_seed(42);m=RawTemporalModel().eval();b=synthetic();r=converged_ig(m,b)
        self.assertLessEqual(abs(r['residual']),.003+.003*abs(r['logit']-r['reference']))

    def test_local_encoder_receptive_fields_cover_every_real_sample(self):
        from diagramo.temporal.encoders import LocalMovement
        m=LocalMovement(1,2).eval()
        with torch.no_grad():
            m.first.weight.fill_(1.);m.first.bias.zero_();m.second.weight.fill_(1.);m.second.bias.zero_()
            for length in (1,2,5,7,8,9,16,31,64,127,128):
                x=torch.zeros(length,1,128);x[torch.arange(length),0,torch.arange(length)]=1.
                z=m(x,torch.full((length,),length))
                self.assertTrue(bool((z>0).all()),'Lost real sample at length '+str(length))

    def test_tcn_gemm_matches_dilated_convolution_outputs_and_all_gradients(self):
        from diagramo.temporal.encoders import TCN
        torch.manual_seed(42);m=TCN(32,.1).eval();x=torch.randn(3,32,32,requires_grad=True);mask=torch.ones(3,32,dtype=torch.bool);mask[1,20:]=False
        reference=x.masked_fill(~mask[:,:,None],0.)
        for layer in m.blocks:reference=(reference+torch.nn.functional.gelu(layer(reference.transpose(1,2)).transpose(1,2))).masked_fill(~mask[:,:,None],0.)
        reference=m.norm(reference).masked_fill(~mask[:,:,None],0.);actual=m(x,mask)
        torch.testing.assert_close(actual,reference,atol=3e-6,rtol=3e-6)
        weight=torch.randn_like(actual);parameters=[x,*m.parameters()]
        a=torch.autograd.grad((actual*weight).sum(),parameters,retain_graph=True)
        b=torch.autograd.grad((reference*weight).sum(),parameters)
        for ga,gb in zip(a,b):torch.testing.assert_close(ga,gb,atol=3e-5,rtol=3e-5)

if __name__=='__main__':unittest.main()
