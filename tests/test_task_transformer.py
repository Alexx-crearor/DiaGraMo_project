"""Structural, mask, preprocessing, raw-transform and attribution tests."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from diagramo.task_transformer import KinematicTaskNet,TaskPreprocessor,deterministic_seed,integrated_gradients,architecture
from diagramo.motor_nuisance import motor_normalize,perturb_record
from task_transformer_common import CONFIG


class TaskTransformerTests(unittest.TestCase):
    def setUp(self):
        deterministic_seed(42)
        torch.set_num_threads(1)

    def test_task_set_equivariance_with_correct_identity_mapping(self):
        x=torch.randn(3,14,74)
        p=torch.randperm(14)
        for task_type in (False,True):
            model=KinematicTaskNet(task_type=task_type,layers=2,pooling='attention').eval()
            with torch.no_grad():
                old=model(x)
                new=model(x[:,p],task_ids=p)
            torch.testing.assert_close(old,new,atol=2e-6,rtol=0)

    def test_absent_tokens_cannot_affect_output_or_attention(self):
        x=torch.randn(3,14,74)
        mask=torch.ones((3,14),dtype=torch.bool);mask[:,[1,5,8]]=False
        model=KinematicTaskNet(task_type=True,pooling='attention').eval()
        changed=x.clone();changed[:,[1,5,8]]=1e6
        a,attention,pool=model(x,mask,return_attention=True)
        b=model(changed,mask)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertTrue((pool[:,[1,5,8]]==0).all())
        self.assertTrue((attention[0][:,:,:,[1,5,8]]==0).all())
        with self.assertRaises(ValueError):model(x,torch.zeros_like(mask))

    def test_no_attention_mlp_and_parameter_bound(self):
        for candidate in CONFIG['candidates']:
            for family in ('task_mlp','transformer'):
                model=KinematicTaskNet(**architecture(candidate,family)).eval()
                _,attention,_=model(torch.randn(2,14,74),return_attention=True)
                self.assertEqual(len(attention),0 if family=='task_mlp' else candidate['layers'])
                self.assertLess(sum(p.numel() for p in model.parameters()),100000)

    def test_scaler_is_train_only_and_preserves_task_coordinates(self):
        rng=np.random.default_rng(2)
        train=rng.normal(size=(10,1036)).astype('float32');train[:,3]=8.
        pre=TaskPreprocessor().fit(train)
        mean=pre.scaler.mean_.copy()
        out=pre.transform(np.full((3,1036),1000,dtype='float32'))
        self.assertEqual(out.shape,(3,14,74))
        np.testing.assert_array_equal(pre.scaler.mean_,mean)
        np.testing.assert_allclose(mean,train.mean(axis=0,dtype=np.float64),atol=1e-12)
        self.assertTrue((pre.transform(train).reshape(10,1036)[:,3]==0).all())

    def record(self):
        t=np.linspace(0,4,101)
        return dict(time=t,x=50+3*np.sin(t),y=100+np.cos(t),pressure=100+20*np.sin(t),
                    tilt=np.ones(101)*20,azimuth=np.ones(101)*30,pen_status=np.arange(101)%2)

    def test_motor_normalization_raw_invariances_and_kept_duration(self):
        r=self.record();z=motor_normalize(r)
        for kind in ('translation','spatial_scale','pressure_scale'):
            transformed=motor_normalize(perturb_record(r,kind))
            for key in z:np.testing.assert_allclose(z[key],transformed[key],atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(z['time'],r['time'])
        np.testing.assert_array_equal(z['pen_status'],r['pen_status'])
        self.assertEqual(r['x'][0],50.)

    def test_timing_sampling_and_pressure_are_well_defined(self):
        r=self.record()
        for kind in CONFIG['nuisances']:
            t=perturb_record(r,kind)
            self.assertTrue(all(np.isfinite(v).all() for v in t.values()))
            self.assertTrue((np.diff(t['time'])>0).all())
        sampled=perturb_record(r,'sampling_half')
        self.assertEqual(sampled['time'][0],r['time'][0]);self.assertEqual(sampled['time'][-1],r['time'][-1])
        stretched=perturb_record(r,'duration_scale')
        self.assertAlmostEqual(stretched['time'][-1]-stretched['time'][0],4.4)
        warped=perturb_record(r,'timing_warp')
        self.assertAlmostEqual(warped['time'][-1]-warped['time'][0],4.)

    def test_integrated_gradients_linear_completeness_and_masks(self):
        class Linear(torch.nn.Module):
            def forward(self,x,present):return (x*present[:,:,None]).sum((1,2))*.1+.3
        X=np.random.default_rng(5).normal(size=(4,14,74)).astype('float32')
        mask=np.ones(14,bool);mask[[2,4]]=False
        ig,base,logits,error=integrated_gradients(Linear(),X,mask,steps=8)
        np.testing.assert_allclose(ig,X*mask[None,:,None]*.1,atol=1e-7)
        self.assertLess(np.max(np.abs(error)),2e-6)
        self.assertTrue((ig[:,[2,4]]==0).all())

    def test_deterministic_initialization(self):
        deterministic_seed(43);a=KinematicTaskNet().state_dict()
        deterministic_seed(43);b=KinematicTaskNet().state_dict()
        for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)

    def test_segmented_ig_resolves_sharp_near_reference_transition(self):
        from analyze_task_transformer import segmented_ig
        class Sharp(torch.nn.Module):
            def forward(self,x,present):return torch.tanh(x[:,0,0])
        X=np.zeros((1,14,74),dtype='float32');X[0,0,0]=1e13
        mask=np.ones(14,bool)
        phi,base,logits,error,panels=segmented_ig(Sharp(),X,mask,'cpu',order=32)
        self.assertLess(abs(error[0]),2e-5)
        self.assertAlmostEqual(phi[0,0,0],1.,places=4)
        self.assertGreater(panels[0],10)


if __name__=='__main__':unittest.main()
