"""Small matched statistical/MIL/relative-stroke families for Phase2.

This module does not select an architecture or fit anything. Relative spatial
coordinates use source device units, not calibrated biomechanical distances.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from diagramo.temporal.encoders import Pool,LocalMovement
from diagramo.temporal_v2.models import LocalReadout
from diagramo.temporal.hierarchical_model import scatter_average
from diagramo.temporal.segmentation import groups


def dilated_conv_gemm(x,conv):
    """Exact dilation cross-correlation; avoids slow variable-bag cuDNN plans."""
    b,d,l=x.shape;dilation=conv.dilation[0]
    padded=torch.nn.functional.pad(x,(dilation,dilation))
    view=torch.stack([padded[:,:,k*dilation:k*dilation+l] for k in range(3)],-1)
    view=view.permute(0,2,1,3).reshape(b*l,d*3)
    return torch.nn.functional.linear(view,conv.weight.reshape(conv.out_channels,-1),conv.bias).reshape(b,l,conv.out_channels).transpose(1,2)


class MultiScaleLocal(nn.Module):
    def __init__(self,dim=32):
        super().__init__();self.kernels=(5,13,29)
        self.branches=nn.ModuleList([nn.Conv1d(21,8,k,stride=4,padding=k//2) for k in self.kernels])
        self.mix=nn.Conv1d(24,dim,1)
        self.dilated=nn.ModuleList([nn.Conv1d(dim,dim,3,padding=d,dilation=d) for d in (2,4)])
    def feature_map(self,x,length):
        mask=torch.arange(x.shape[-1],device=x.device)[None,:]<length[:,None]
        x=torch.nn.functional.pad(x.masked_fill(~mask[:,None,:],0.),(0,16))
        values=[];union=None
        for kernel,conv in zip(self.kernels,self.branches):
            z=torch.nn.functional.gelu(conv(x));valid=torch.arange(z.shape[-1],device=x.device)[None,:]*4-kernel//2<length[:,None]
            values.append(z.masked_fill(~valid[:,None,:],0.));union=valid if union is None else union|valid
        z=torch.nn.functional.gelu(self.mix(torch.cat(values,1))).masked_fill(~union[:,None,:],0.)
        for conv in self.dilated:
            z=(z+torch.nn.functional.gelu(dilated_conv_gemm(z,conv))).masked_fill(~union[:,None,:],0.)
        return z,union
    def forward(self,x,length):
        z,valid=self.feature_map(x,length);tokens=[];present=[]
        for indices in torch.tensor_split(torch.arange(z.shape[-1],device=x.device),4):
            m=valid[:,indices];n=m.sum(1)
            tokens.append(z[:,:,indices].sum(-1)/n.clamp_min(1)[:,None]);present.append(n>0)
        return torch.stack(tokens,1),torch.stack(present,1)


class RelativeStrokeBlock(nn.Module):
    def __init__(self,dim=32,heads=2,dropout=.1,time=True,spatial=True,index_only=False):
        super().__init__();self.heads=heads;self.use_time=time;self.use_spatial=spatial;self.index_only=index_only
        self.norm=nn.LayerNorm(dim);self.qkv=nn.Linear(dim,3*dim);self.output=nn.Linear(dim,dim)
        self.bias=nn.Sequential(nn.Linear(7,8),nn.Tanh(),nn.Linear(8,heads))
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,2*dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*dim,dim))
        self.drop=nn.Dropout(dropout)
    def pair_features(self,meta):
        # [start,end,duration,centre_x,centre_y,pen_down_fraction,ordinal]
        start=meta[:,:,6] if self.index_only else meta[:,:,0]
        duration=torch.ones_like(meta[:,:,2]) if self.index_only else meta[:,:,2].clamp_min(1e-6)
        dt=start[:,:,None]-start[:,None,:]
        dx=meta[:,:,None,3]-meta[:,None,:,3];dy=meta[:,:,None,4]-meta[:,None,:,4]
        def slog(a):return a.sign()*torch.log1p(a.abs())/4
        elapsed=slog(dt) if self.use_time else torch.zeros_like(dt)
        gap=slog((start[:,:,None]-(start+duration)[:,None,:]).clamp_min(0)) if self.use_time else torch.zeros_like(dt)
        ratio=torch.log(duration[:,:,None]/duration[:,None,:]).clamp(-10,10)/4 if self.use_time else torch.zeros_like(dt)
        return torch.stack([elapsed,gap,ratio,slog(dx) if self.use_spatial else dx*0,
            slog(dy) if self.use_spatial else dy*0,
            meta[:,:,None,5]-meta[:,None,:,5],
            torch.eye(meta.shape[1],device=meta.device,dtype=meta.dtype)[None].expand(meta.shape[0],-1,-1)],-1)
    def forward(self,x,mask,meta):
        b,l,d=x.shape;z=self.qkv(self.norm(x)).reshape(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        bias=self.bias(self.pair_features(meta)).permute(0,3,1,2)
        scores=z[0]@z[1].transpose(-1,-2)/math.sqrt(d//self.heads)+bias
        safe=mask.clone();safe[:,0]|=~safe.any(1)
        a=scores.masked_fill(~safe[:,None,None,:],-1e9).softmax(-1)*mask[:,None,None,:]
        h=(self.drop(a)@z[2]).transpose(1,2).reshape(b,l,d)
        x=x+self.drop(self.output(h));x=x+self.drop(self.ff(x))
        return x.masked_fill(~mask[:,:,None],0.),a


class OriginalRetainedLocal(nn.Module):
    def __init__(self,dim=32):
        super().__init__();self.readout=LocalReadout(LocalMovement(21,dim),'multi4',dim)
    def feature_map(self,x,length):return self.readout.feature_map(x,length)
    def forward(self,x,length):return self.readout.encode(x,length)


class MultiScaleModel(nn.Module):
    def __init__(self,design='mil',relative_time=True,relative_spatial=True,index_only=False,dropout=.1,d_model=32,local_encoder='multiscale',local_chunk_windows=256):
        super().__init__();assert design in ('statistical','mil','stroke_transformer')
        self.config=dict(design=design,relative_time=relative_time,relative_spatial=relative_spatial,
            index_only=index_only,dropout=dropout,d_model=d_model,local_encoder=local_encoder,local_chunk_windows=local_chunk_windows)
        self.local_chunk_windows=local_chunk_windows
        assert local_encoder in ('original','multiscale')
        self.family='cnn';self.design=design;self.local=MultiScaleLocal(d_model) if local_encoder=='multiscale' else OriginalRetainedLocal(d_model);self.pool=Pool(d_model)
        self.statistics=nn.Sequential(nn.Linear(3*d_model,d_model),nn.GELU()) if design=='statistical' else None
        self.context=RelativeStrokeBlock(d_model,2,dropout,relative_time,relative_spatial,index_only) if design=='stroke_transformer' else None
        self.task_id=nn.Embedding(14,d_model);self.task_ff=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout))
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,1));self.drop=nn.Dropout(dropout)
    def encode_local(self,x,length):
        tokens=[];presence=[]
        for start in range(0,len(length),self.local_chunk_windows):
            at=slice(start,start+self.local_chunk_windows)
            if torch.is_grad_enabled() and (self.training or x.requires_grad):
                z,valid=checkpoint(self.local,x[at],length[at],use_reentrant=False,preserve_rng_state=False)
            else:z,valid=self.local(x[at],length[at])
            tokens.append(z);presence.append(valid)
        return torch.cat(tokens),torch.cat(presence)
    def aggregate(self,x,mask):
        if self.statistics is None:return self.pool(x,mask)
        n=mask.sum(1).clamp_min(1)[:,None];mean=(x*mask[:,:,None]).sum(1)/n
        maximum=x.masked_fill(~mask[:,:,None],-torch.inf).amax(1);maximum=torch.where(mask.any(1)[:,None],maximum,0.)
        std=torch.sqrt(((x-mean[:,None,:]).square()*mask[:,:,None]).sum(1)/n+1e-8)
        z=self.statistics(torch.cat([mean,maximum,std],1));return z*mask.any(1)[:,None],mask/n
    def forward(self,batch,patch_present=None,segment_present=None,return_details=False,stress=None):
        device=batch['x'].device;n=batch.get('person_count',1);d=self.head[-1].in_features
        tokens,valid=self.encode_local(batch['x'],batch['length']);p,k,_=tokens.shape
        if patch_present is not None:valid&=patch_present[:,None]
        z=self.drop(tokens.reshape(-1,d));present=valid.flatten();segmentid=batch['patch_segment'].repeat_interleave(k)
        indices=torch.as_tensor(groups(segmentid.detach().cpu().numpy(),16*k),device=device)
        safe=indices.clamp_min(0);mask=(indices>=0)&present[safe]
        chunks,window_weights=self.aggregate(z[safe].masked_fill(~mask[:,:,None],0.),mask)
        segment,active=scatter_average(chunks,segmentid[indices[:,0]],mask.sum(1).to(z.dtype),len(batch['segment_task']))
        if segment_present is not None:active&=segment_present
        indices=batch['task_groups'];safe=indices.clamp_min(0);mask=(indices>=0)&active[safe]
        stroke=segment[safe].masked_fill(~mask[:,:,None],0.);meta=batch['segment_relation'][safe]
        if stress:
            # Group-wise stress keeps absent padding out of the permutation.
            stroke=stroke.clone();meta=meta.clone()
            for row in range(len(stroke)):
                at=torch.nonzero(mask[row]).flatten();order=at.flip(0)
                stroke[row,at]=stroke[row,order].clone()
                if stress=='transport':meta[row,at]=meta[row,order].clone()
                else:assert stress=='content_only','Unknown stress policy'
        attention=None
        if self.context is not None:stroke,attention=self.context(stroke,mask,meta)
        chunks,stroke_weights=self.aggregate(stroke,mask)
        task,task_valid=scatter_average(chunks,batch['task_group_task'],mask.sum(1).to(z.dtype),14*n)
        task=self.task_ff(task+self.task_id(torch.arange(14*n,device=device)%14)).reshape(n,14,d)
        person,task_weights=self.aggregate(task,task_valid.reshape(n,14));logit=self.head(person).reshape(n)
        details=dict(window_tokens=tokens,window_token_valid=valid,segment=segment,segment_present=active,
            task=task,task_valid=task_valid.reshape(n,14),person=person,window_pool=window_weights,
            stroke_pool=stroke_weights,participant_pool=task_weights,relative_attention=attention)
        return (logit,details) if return_details else logit
