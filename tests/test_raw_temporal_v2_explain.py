import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from raw_temporal_v2_explain import integrated_motor_gradients,motor_reference,cached_local,randomized_head,FrozenReadoutTorch
from diagramo.temporal_v2.multiscale import MultiScaleModel,OriginalRetainedLocal
from test_raw_temporal_v2_models import bag


class LinearScore(torch.nn.Module):
    def forward(self,batch):
        return (batch['x']*torch.arange(21)[None,:,None]).sum().reshape(1)+3


def test_ig_linear_closed_form_raw_logit_completeness():
    torch.manual_seed(2);x=torch.randn(2,21,128);x[:,7]=1.;x[:,13:15]=1.
    batch=dict(x=x,sample_mask=torch.ones(2,128,dtype=torch.bool))
    reference=motor_reference(batch,'train_mean');model=LinearScore()
    attribution,result=integrated_motor_gradients(model,batch,reference,steps=(4,),atol=.001,rtol=.001)
    expected=(x-reference)*torch.arange(21)[None,:,None]
    torch.testing.assert_close(attribution,expected);assert result['complete']
    assert not attribution[:,5:8].any() and not attribution[:,13:].any()


def test_window_reference_preserves_invalid_motion_and_clock():
    torch.manual_seed(1);x=torch.randn(3,21,128);x[:,7]=0.;x[:,13:15]=0.
    batch=dict(x=x,sample_mask=torch.ones(3,128,dtype=torch.bool))
    reference=motor_reference(batch,'window_mean')
    torch.testing.assert_close(reference[:,8:13],x[:,8:13]);torch.testing.assert_close(reference[:,2],x[:,2])
    torch.testing.assert_close(reference[:,5:8],x[:,5:8])


def test_local_cache_does_not_accumulate_masking_and_head_restores():
    torch.manual_seed(42);torch.set_num_threads(2);model=MultiScaleModel('mil',dropout=0).eval();batch=bag()
    batch['segment_relation']=torch.zeros(28,7)
    with torch.no_grad():baseline=model(batch).clone()
    with cached_local(model,batch):
        present=torch.ones(28,dtype=torch.bool);present[:8]=False
        with torch.no_grad():
            first=model(batch,patch_present=present);second=model(batch,patch_present=present)
            torch.testing.assert_close(first,second);torch.testing.assert_close(model(batch),baseline)
    with randomized_head(model,2026):
        with torch.no_grad():assert (model(batch)-baseline).abs().max()>1e-6
    with torch.no_grad():torch.testing.assert_close(model(batch),baseline)


@pytest.mark.parametrize('kind',['linear','nonlinear'])
def test_frozen_torch_readout_matches_independent_numpy_sklearn(kind):
    torch.manual_seed(5);torch.set_num_threads(2);encoder=OriginalRetainedLocal().eval();batch=bag()
    with torch.no_grad():tokens,valid=encoder(batch['x'],batch['length'])
    tokens=tokens.numpy();valid=valid.numpy();task=batch['patch_task'].numpy();parts=[]
    for t in range(14):
        selected=tokens[valid&(task==t)[:,None]]
        parts.extend([np.mean(selected,axis=0),np.max(selected,axis=0),np.std(selected,axis=0)])
    features=np.concatenate(parts).astype('float32')[None]
    rng=np.random.default_rng(1);train=rng.normal(size=(20,1344)).astype('float32');labels=np.asarray([0,1]*10)
    estimator=LogisticRegression(C=.01,max_iter=200) if kind=='linear' else MLPClassifier(hidden_layer_sizes=(4,),activation='tanh',alpha=1.,solver='lbfgs',max_iter=200,random_state=1)
    pipeline=make_pipeline(StandardScaler(),estimator).fit(train,labels)
    model=FrozenReadoutTorch(dict(local_encoder='original',local_state=encoder.state_dict()),pipeline).eval()
    with torch.no_grad():probability=model(batch).sigmoid().numpy()
    np.testing.assert_allclose(probability,pipeline.predict_proba(features)[:,1],atol=1e-5,rtol=1e-5)
    batch['x']=batch['x'].requires_grad_(True);gradient=torch.autograd.grad(model(batch).sum(),batch['x'])[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
