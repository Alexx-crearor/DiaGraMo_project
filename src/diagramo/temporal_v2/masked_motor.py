"""Conservative masked motor objective; no participant labels or pooled targets.

Mask original positions and dependent velocity/acceleration channels together,
including a two-sample halo. Clock, pen-state and validity flags remain visible.
Targets use normalized signed-log source coordinates, pressure and speed.
"""
import torch
from torch import nn

TARGET_CHANNELS=(0,1,2,10)
HIDDEN_CHANNELS=(0,1,2,3,4,8,9,10,11,12)


def fixed_stride_linear_upsample(prediction,stride):
    """Linear interpolation at integer-stride centres with deterministic GEMM.

    Equivalent to align_corners=True with output length (L-1)*stride+1.
    CUDA's interpolate backward uses atomic accumulation; this fixed linear
    operator instead uses matrix multiplication in both forward and backward.
    """
    assert isinstance(stride,int) and stride>0
    centres=prediction.shape[-1]
    positions=torch.arange((centres-1)*stride+1,device=prediction.device)
    distance=(positions[None,:]-torch.arange(centres,device=prediction.device)[:,None]*stride).abs()
    weights=(stride-distance).clamp_min(0).to(prediction.dtype)/stride
    return prediction@weights


def corrupt(x,length,generator,span=32,halo=2):
    """Return corrupted input and target masks. Mask choice never reads values."""
    b,c,l=x.shape;sample=torch.arange(l,device=x.device)[None,:]<length[:,None]
    pen=sample&(x[:,7,:]>.5);eligible=pen.any(1)
    # Uniform pen-down anchor without per-window CUDA synchronization.
    choice=torch.rand(b,generator=generator,device=x.device)*pen.sum(1)
    center=(pen.cumsum(1)<=choice[:,None]).sum(1).clamp_max(l-1)
    width=length.clamp_max(span);first=torch.minimum((center-width//2).clamp_min(0),length-width)
    positions=torch.arange(l,device=x.device)[None,:]
    target=(positions>=first[:,None])&(positions<(first+width)[:,None])&eligible[:,None]&sample
    hidden=(positions>=(first-halo)[:,None])&(positions<(first+width+halo)[:,None])&eligible[:,None]&sample
    corrupted=x.clone()
    for channel in HIDDEN_CHANNELS:corrupted[:,channel,:]=corrupted[:,channel,:].masked_fill(hidden,0.)
    mask=(target&pen)[:,None,:].expand(-1,len(TARGET_CHANNELS),-1).clone()
    mask[:,3,:]&=x[:,13,:]>.5
    return corrupted,mask,hidden


def window_loss(prediction,target,mask):
    count=mask.sum(-1);defined=count>0
    errors=((prediction-target).square()*mask).sum(-1)/count.clamp_min(1)
    per_window=(errors*defined).sum(1)/defined.sum(1).clamp_min(1)
    return per_window,defined.any(1)


class MaskedMotorModel(nn.Module):
    def __init__(self,local,stride=4,dim=32):
        super().__init__();self.local=local;self.stride=stride
        self.decoder=nn.Sequential(nn.Conv1d(dim,dim,3,padding=1),nn.GELU(),nn.Conv1d(dim,len(TARGET_CHANNELS),1))
    def forward(self,x,length):
        z,valid=self.local.feature_map(x,length);z=z.masked_fill(~valid[:,None,:],0.)
        prediction=self.decoder(z)
        # Map feature centres j*stride to source sample positions without
        # compressing the explicit right-hand receptive-field coverage padding.
        prediction=fixed_stride_linear_upsample(prediction,self.stride)
        assert prediction.shape[-1]>=x.shape[-1]
        return prediction[:,:,:x.shape[-1]]
