import sys
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import pytest
from diagramo.temporal_v2.masked_motor import fixed_stride_linear_upsample
from diagramo.temporal_v2.masked_motor import corrupt,window_loss,MaskedMotorModel,TARGET_CHANNELS,HIDDEN_CHANNELS
from diagramo.temporal_v2.multiscale import MultiScaleLocal,OriginalRetainedLocal


def inputs():
    torch.manual_seed(1);x=torch.randn(4,21,128);x[:,7]=1.;x[:,13]=1.
    length=torch.tensor([128,67,1,128]);x[3,7]=0.;x[1,:,67:]=0.;x[2,:,1:]=0.
    return x,length


def test_no_copied_targets_or_halo_derivatives():
    x,length=inputs();a,mask,hidden=corrupt(x,length,torch.Generator().manual_seed(42))
    changed=x.clone()
    for channel in HIDDEN_CHANNELS:changed[:,channel,:]=changed[:,channel,:]+hidden*100
    b,other,_=corrupt(changed,length,torch.Generator().manual_seed(42))
    torch.testing.assert_close(a,b);assert torch.equal(mask,other)
    assert not mask[3].any() and not mask[1,:,67:].any() and not mask[2,:,1:].any()


def test_invalid_derivative_and_padding_have_zero_objective():
    x,length=inputs();x[:,13]=0.
    a,mask,hidden=corrupt(x,length,torch.Generator().manual_seed(42));assert not mask[:,3].any()
    target=x[:,TARGET_CHANNELS,:];prediction=target.clone();prediction[:,3]=1e6
    loss,defined=window_loss(prediction,target,mask);assert not loss.any() and not defined[3]


def test_halo_removes_recomputed_velocity_acceleration_shortcut():
    torch.manual_seed(3);coordinate=torch.randn(128).cumsum(0)
    def signal(position):
        x=torch.zeros(1,21,128);x[0,0]=position
        velocity=torch.cat([torch.zeros(1),position.diff()])
        acceleration=torch.cat([torch.zeros(1),velocity.diff()])
        x[0,8]=velocity;x[0,10]=velocity.abs();x[0,11]=acceleration
        x[:,7]=1.;x[:,13]=1.;x[:,14]=1.;return x
    length=torch.tensor([128]);a=signal(coordinate)
    _,mask,_=corrupt(a,length,torch.Generator().manual_seed(0),halo=2)
    changed=coordinate.clone();changed[mask[0,0]]+=100
    b=signal(changed)
    masked_a,_,_=corrupt(a,length,torch.Generator().manual_seed(0),halo=2)
    masked_b,_,_=corrupt(b,length,torch.Generator().manual_seed(0),halo=2)
    torch.testing.assert_close(masked_a,masked_b)
    unprotected_a,_,_=corrupt(a,length,torch.Generator().manual_seed(0),halo=0)
    unprotected_b,_,_=corrupt(b,length,torch.Generator().manual_seed(0),halo=0)
    assert (unprotected_a-unprotected_b).abs().max()>1


def test_both_encoder_decoders_cover_original_tail_and_backpropagate():
    torch.set_num_threads(2);x,length=inputs();masked,mask,_=corrupt(x,length,torch.Generator().manual_seed(42))
    for local,stride in [(MultiScaleLocal(),4),(OriginalRetainedLocal(),8)]:
        model=MaskedMotorModel(local,stride);prediction=model(masked,length);assert prediction.shape==(4,4,128)
        loss,defined=window_loss(prediction,x[:,TARGET_CHANNELS,:],mask)
        loss[defined].mean().backward();assert torch.isfinite(loss).all()
        assert sum(float(p.grad.abs().sum()) for p in local.parameters() if p.grad is not None)>0


@pytest.mark.parametrize('stride,centres', [(4,33),(8,17),(1,5)])
def test_fixed_stride_interpolation_matches_values_and_gradients(stride,centres):
    torch.manual_seed(11)
    x=torch.randn(3,4,centres,dtype=torch.float64,requires_grad=True)
    reference=torch.nn.functional.interpolate(x,size=(centres-1)*stride+1,mode='linear',align_corners=True)
    result=fixed_stride_linear_upsample(x,stride)
    torch.testing.assert_close(result,reference,atol=1e-12,rtol=1e-12)
    upstream=torch.randn_like(reference)
    expected=torch.autograd.grad(reference,x,upstream)[0]
    actual=torch.autograd.grad(result,x,upstream)[0]
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA regression check')
def test_cuda_ssl_decoder_backward_is_strictly_deterministic():
    previous=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(42);x,length=inputs();x=x.cuda();length=length.cuda()
        masked,mask,_=corrupt(x,length,torch.Generator(device='cuda').manual_seed(42))
        for local,stride in [(MultiScaleLocal(),4),(OriginalRetainedLocal(),8)]:
            model=MaskedMotorModel(local,stride).cuda();gradients=[]
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                prediction=model(masked,length)
                loss,defined=window_loss(prediction,x[:,TARGET_CHANNELS,:],mask)
                loss[defined].mean().backward()
                gradients.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
                assert torch.isfinite(gradients[-1]).all()
            assert torch.equal(gradients[0],gradients[1])
    finally:torch.use_deterministic_algorithms(previous)
