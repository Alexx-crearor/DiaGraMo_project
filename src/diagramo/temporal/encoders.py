"""Small masked local, dilated convolution and bounded temporal attention blocks."""
import math
import torch
from torch import nn

class LocalMovement(nn.Module):
    def __init__(self,channels,dim):
        super().__init__();self.first=nn.Conv1d(channels,8,9,stride=8,padding=4);self.second=nn.Conv1d(8,dim,3,padding=1);self.act=nn.GELU()
    def forward(self,x,length):
        m=torch.arange(x.shape[-1],device=x.device)[None,:]<length[:,None]
        x=x.masked_fill(~m[:,None,:],0.)
        # Retain every output whose receptive field overlaps a real sample,
        # including the final partial stride. Otherwise last1–3 samples vanish.
        y=self.act(self.first(torch.nn.functional.pad(x,(0,8))))
        valid=torch.arange(y.shape[-1],device=x.device)[None,:]*8-4<length[:,None]
        y=self.act(self.second(y.masked_fill(~valid[:,None,:],0.)))
        return (y*valid[:,None,:]).sum(-1)/valid.sum(-1).clamp_min(1)[:,None]

class Pool(nn.Module):
    def __init__(self,dim):super().__init__();self.score=nn.Linear(dim,1)
    def forward(self,x,mask):
        safe=mask.clone();safe[:,0]|=~safe.any(1)
        w=self.score(x).squeeze(-1).masked_fill(~safe,-1e9).softmax(-1)*mask
        return (x*w[:,:,None]).sum(1),w

class TemporalBlock(nn.Module):
    def __init__(self,dim,dropout,heads=2):
        super().__init__();self.heads=heads;self.norm=nn.LayerNorm(dim);self.qkv=nn.Linear(dim,3*dim);self.out=nn.Linear(dim,dim)
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,2*dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*dim,dim));self.drop=nn.Dropout(dropout)
    def forward(self,x,mask):
        B,L,D=x.shape;z=self.qkv(self.norm(x)).reshape(B,L,3,self.heads,D//self.heads).permute(2,0,3,1,4)
        safe=mask.clone();safe[:,0]|=~safe.any(1)
        a=(z[0]@z[1].transpose(-1,-2))/math.sqrt(D//self.heads);a=a.masked_fill(~safe[:,None,None,:],-1e9).softmax(-1)
        a=a*mask[:,None,None,:];h=(self.drop(a)@z[2]).transpose(1,2).reshape(B,L,D)
        x=x+self.drop(self.out(h));x=x+self.drop(self.ff(x))
        return x.masked_fill(~mask[:,:,None],0.),a

class TCN(nn.Module):
    def __init__(self,dim,dropout):
        super().__init__();self.blocks=nn.ModuleList([nn.Conv1d(dim,dim,3,padding=d,dilation=d)for d in (1,2,4)]);self.drop=nn.Dropout(dropout);self.norm=nn.LayerNorm(dim)
    def forward(self,x,mask):
        x=x.masked_fill(~mask[:,:,None],0.)
        for layer in self.blocks:
            # Exact dilated cross-correlation, using GEMM to avoid very slow
            # variable-batch cuDNN Conv1d plans on the T1000/Windows backend.
            B,L,D=x.shape;d=layer.dilation[0]
            padded=torch.nn.functional.pad(x.transpose(1,2),(d,d))
            view=torch.stack([padded[:,:,k*d:k*d+L]for k in range(3)],dim=-1)
            view=view.permute(0,2,1,3).reshape(B*L,D*3)
            value=torch.nn.functional.linear(view,layer.weight.reshape(D,-1),layer.bias).reshape(B,L,D)
            x=(x+self.drop(torch.nn.functional.gelu(value))).masked_fill(~mask[:,:,None],0.)
        return self.norm(x).masked_fill(~mask[:,:,None],0.)

