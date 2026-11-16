"""Local movement -> bounded stroke -> bounded task -> unordered participant."""
import torch
from torch import nn
from .encoders import LocalMovement,TemporalBlock,TCN,Pool
from .preprocessing import CHANNELS

def scatter_average(x,index,weight,count):
    result=x.new_zeros((count,x.shape[-1]));den=x.new_zeros(count)
    result.index_add_(0,index,x*weight[:,None]);den.index_add_(0,index,weight)
    return result/den.clamp_min(1e-8)[:,None],den>0

class RawTemporalModel(nn.Module):
    def __init__(self,family='hierarchical',d_model=32,dropout=.1):
        super().__init__();self.family=family;self.config=dict(family=family,d_model=d_model,dropout=dropout)
        self.local=LocalMovement(len(CHANNELS),d_model);self.time=nn.Sequential(nn.Linear(4,d_model),nn.Tanh())
        self.pool=Pool(d_model);self.drop=nn.Dropout(dropout);self.task_id=nn.Embedding(14,d_model)
        self.hier=family not in ('cnn','tcn','transformer')
        self.stroke=TemporalBlock(d_model,dropout)if self.hier else None
        self.task_context=TemporalBlock(d_model,dropout)if family in ('transformer','hierarchical','no_explicit_time','no_position','index_time')else None
        self.tcn=TCN(d_model,dropout)if family=='tcn'else None
        self.person_context=TemporalBlock(d_model,dropout)if self.hier and family!='local_stroke'else None
        self.task_ff=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout))
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,1))

    def grouped(self,x,present,indices,context=None):
        mask=indices>=0;safe=indices.clamp_min(0);mask=mask&present[safe]
        tokens=x[safe].masked_fill(~mask[:,:,None],0.)
        if isinstance(context,TemporalBlock):tokens,_=context(tokens,mask)
        elif context is not None:tokens=context(tokens,mask)
        value,weights=self.pool(tokens,mask)
        return value,mask.sum(1).to(x.dtype),weights

    def classify_segments(self,batch,segments,segment_present,task_present=None,segment_order=None,return_details=False):
        """Vectorized held-out segment occlusion; same computation as a full forward."""
        if segment_present.ndim==1:segment_present=segment_present[None]
        K,S=segment_present.shape;device=segments.device;N=batch.get('person_count',1);T=14*N
        tasks=torch.ones((K,T),device=device,dtype=torch.bool)if task_present is None else task_present
        if tasks.ndim==1:tasks=tasks[None].expand(K,-1)
        if segment_order is not None:
            segments=segments[segment_order];segment_present=segment_present[:,segment_order]
        present=segment_present&tasks[:,batch['segment_task']]
        seg=segments
        if self.family not in ('no_explicit_time','no_position'):seg=seg+self.time(batch['segment_meta'])
        indices=batch['task_groups'];safe=indices.clamp_min(0);G,L=indices.shape;D=seg.shape[-1]
        mask=(indices[None]>=0)&present[:,safe]
        tokens=seg[safe][None].expand(K,-1,-1,-1).reshape(K*G,L,D).masked_fill(~mask.reshape(K*G,L,1),0.)
        flatmask=mask.reshape(K*G,L)
        if self.task_context is not None:tokens,_=self.task_context(tokens,flatmask)
        chunks,weights=self.pool(tokens,flatmask);count=flatmask.sum(1).to(segments.dtype)
        destination=(batch['task_group_task'][None]+T*torch.arange(K,device=device)[:,None]).reshape(-1)
        task,valid=scatter_average(chunks,destination,count,K*T);task=task.reshape(K,T,D);valid=(valid.reshape(K,T)&tasks).reshape(K*N,14)
        task=self.task_ff(task+self.task_id(torch.arange(T,device=device)%14)[None]).reshape(K*N,14,D)
        details={'task_pool':weights}
        if self.person_context is not None:task,a=self.person_context(task,valid);details['task_attention']=a
        person,w=self.pool(task,valid);details['participant_pool']=w;logit=self.head(person).reshape(K*N)
        return (logit,details)if return_details else logit

    def forward(self,batch,patch_present=None,task_present=None,return_details=False,local_tokens=None,segment_order=None,return_segments=False):
        x=batch['x'];P=x.shape[0];device=x.device
        present=torch.ones(P,dtype=torch.bool,device=device)if patch_present is None else patch_present
        N=batch.get('person_count',1);T=14*N
        tasks=torch.ones(T,dtype=torch.bool,device=device)if task_present is None else task_present
        present=present&tasks[batch['patch_task']]
        z=self.drop(self.local(x,batch['length']))if local_tokens is None else local_tokens
        z=z.masked_fill(~present[:,None],0.)
        timed=self.family not in ('no_explicit_time','no_position')
        if timed:z=z+self.time(batch['patch_meta'])*present[:,None]
        details={}
        if self.hier:
            chunks,count,w=self.grouped(z,present,batch['stroke_groups'],self.stroke)
            seg,seg_present=scatter_average(chunks,batch['stroke_group_segment'],count,len(batch['segment_task']))
            if return_segments:return seg,seg_present
            logit,details=self.classify_segments(batch,seg,seg_present,tasks,segment_order,return_details=True)
            details['stroke_pool']=w
            logit=logit.reshape(N)
            return (logit,details)if return_details else logit
        else:
            chunks,count,w=self.grouped(z,present,batch['flat_groups'],self.task_context if self.task_context is not None else self.tcn)
            task,task_valid=scatter_average(chunks,batch['flat_group_task'],count,T);details['flat_pool']=w
        task_valid=task_valid&tasks
        task=self.task_ff(task+self.task_id(torch.arange(T,device=device)%14)).reshape(N,14,-1);task_valid=task_valid.reshape(N,14)
        person,weights=self.pool(task,task_valid);details['participant_pool']=weights
        logit=self.head(person).reshape(N)
        return (logit,details)if return_details else logit
