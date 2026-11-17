"""Relative geometry is permutation equivariant; masks never become observations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import pytest
from diagramo.temporal_v2.multiscale import MultiScaleLocal,RelativeStrokeBlock,MultiScaleModel,dilated_conv_gemm
from test_raw_temporal_v2_models import bag


def test_gemm_dilation_matches_convolution_and_input_gradient():
    torch.manual_seed(42);torch.set_num_threads(2)
    for dilation in (2,4):
        layer=torch.nn.Conv1d(32,32,3,padding=dilation,dilation=dilation)
        x=torch.randn(5,32,36,requires_grad=True)
        reference=layer(x);actual=dilated_conv_gemm(x,layer)
        torch.testing.assert_close(reference,actual,atol=2e-6,rtol=2e-6)
        a=torch.autograd.grad(reference.square().sum(),x,retain_graph=True)[0]
        b=torch.autograd.grad(actual.square().sum(),x)[0]
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)


def test_multiscale_padding_and_tail_coverage():
    torch.manual_seed(42);torch.set_num_threads(2);model=MultiScaleLocal().eval()
    x=torch.randn(3,21,128);length=torch.tensor([1,61,128]);other=x.clone()
    for i,n in enumerate(length):other[i,:,n:]=1000
    a,mask=model(x,length);b,_=model(other,length);torch.testing.assert_close(a,b)
    changed=x.clone();changed[-1,:,-1]+=10;c,_=model(changed,length)
    assert (a[-1]-c[-1]).abs().max()>1e-5
    a.sum().backward();assert all(conv.weight.grad.abs().sum()>0 for conv in model.branches)


def test_chunk_checkpoint_matches_full_encoder_and_gradient():
    torch.manual_seed(42);torch.set_num_threads(2)
    model=MultiScaleModel(local_chunk_windows=3).train();x=torch.randn(8,21,128);length=torch.tensor([128,33,5,128,91,64,1,17])
    a,valid=model.local(x,length);b,other=model.encode_local(x,length)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6);assert torch.equal(valid,other)
    parameter=model.local.branches[0].weight
    ga=torch.autograd.grad(a.square().sum(),parameter,retain_graph=True)[0]
    gb=torch.autograd.grad(b.square().sum(),parameter)[0]
    torch.testing.assert_close(ga,gb,atol=2e-5,rtol=2e-5)


def test_relative_attention_transport_equivariance():
    torch.manual_seed(42);model=RelativeStrokeBlock(dropout=0).eval()
    x=torch.randn(2,9,32);mask=torch.ones(2,9,dtype=torch.bool);mask[1,6:]=False
    meta=torch.randn(2,9,7);meta[:,:,2]=meta[:,:,2].abs()+.01
    order=torch.tensor([4,2,0,8,7,1,5,6,3]);inverse=torch.argsort(order)
    a,_=model(x,mask,meta);b,_=model(x[:,order],mask[:,order],meta[:,order])
    torch.testing.assert_close(a,b[:,inverse],atol=2e-6,rtol=2e-6)
    mask[:]=False;c,_=model(x,mask,meta);assert torch.isfinite(c).all() and not c.any()


def test_relative_feature_ablation_isolation():
    meta=torch.randn(2,5,7);meta[:,:,2].abs_().add_(.1)
    changed=meta.clone();changed[:,:,:3]=changed[:,:,:3]*99+33
    no_time=RelativeStrokeBlock(time=False)
    torch.testing.assert_close(no_time.pair_features(meta),no_time.pair_features(changed))
    index=RelativeStrokeBlock(index_only=True)
    torch.testing.assert_close(index.pair_features(meta),index.pair_features(changed))
    changed=meta.clone();changed[:,:,3:5]+=torch.randn(2,5,2)*100
    no_space=RelativeStrokeBlock(spatial=False)
    torch.testing.assert_close(no_space.pair_features(meta),no_space.pair_features(changed))


@pytest.mark.parametrize('design',['statistical','mil','stroke_transformer'])
def test_multiscale_models_transport_invariance(design):
    torch.manual_seed(42);model=MultiScaleModel(design,dropout=0).eval();b=bag()
    b['segment_relation']=torch.randn(len(b['segment_task']),7);b['segment_relation'][:,2].abs_().add_(.01)
    a=model(b);c=model(b,stress='transport');torch.testing.assert_close(a,c,atol=2e-6,rtol=2e-6)
    a.sum().backward();assert model.local.branches[0].weight.grad.abs().sum()>0
