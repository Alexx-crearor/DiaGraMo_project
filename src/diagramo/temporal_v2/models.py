"""Matched local readouts and independently controlled hierarchy aggregation."""
import numpy as np
import torch
from torch import nn
from diagramo.temporal.hierarchical_model import RawTemporalModel,scatter_average
from diagramo.temporal.encoders import TemporalBlock
from diagramo.temporal.segmentation import groups


class LocalReadout(nn.Module):
    def __init__(self,backbone,mode='mean',dim=32):
        super().__init__();self.backbone=backbone;self.mode=mode;self.dim=dim
        self.score=nn.Linear(dim,1) if mode=='attention' else None
        if self.score is not None:nn.init.zeros_(self.score.weight);nn.init.zeros_(self.score.bias)
        width={'mean_max':2,'mean_max_std':3}.get(mode)
        self.projection=nn.Linear(width*dim,dim) if width else None
        if self.projection is not None:
            # Matched initial mean representation, extra statistics start at0.
            nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
            with torch.no_grad():self.projection.weight[:,:dim].copy_(torch.eye(dim))

    def feature_map(self,x,length):
        backbone=self.backbone
        mask=torch.arange(x.shape[-1],device=x.device)[None,:]<length[:,None]
        values=backbone.act(backbone.first(torch.nn.functional.pad(x.masked_fill(~mask[:,None,:],0.),(0,8))))
        valid=torch.arange(values.shape[-1],device=x.device)[None,:]*8-4<length[:,None]
        values=backbone.act(backbone.second(values.masked_fill(~valid[:,None,:],0.)))
        return values.masked_fill(~valid[:,None,:],0.),valid

    def encode(self,x,length):
        values,valid=self.feature_map(x,length);count=valid.sum(-1).clamp_min(1)
        mean=values.sum(-1)/count[:,None]
        if self.mode=='multi4':
            outputs=[];presence=[]
            for indices in torch.tensor_split(torch.arange(values.shape[-1],device=x.device),4):
                active=valid[:,indices];number=active.sum(-1)
                outputs.append(values[:,:,indices].sum(-1)/number.clamp_min(1)[:,None]);presence.append(number>0)
            return torch.stack(outputs,1),torch.stack(presence,1)
        if self.mode in ('mean_max','mean_max_std'):
            maximum=values.masked_fill(~valid[:,None,:],-torch.inf).amax(-1)
            maximum=torch.where(valid.any(-1)[:,None],maximum,0.)
            pieces=[mean,maximum]
            if self.mode=='mean_max_std':
                variance=((values-mean[:,:,None]).square()*valid[:,None,:]).sum(-1)/count[:,None]
                pieces.append(torch.sqrt(variance+1e-8))
            output=self.projection(torch.cat(pieces,-1))
        elif self.mode=='attention':
            scores=self.score(values.transpose(1,2)).squeeze(-1).masked_fill(~valid,-1e9)
            weights=scores.softmax(-1)*valid
            output=(values*weights[:,None,:]).sum(-1)
        else:
            assert self.mode=='mean';output=mean
        return output[:,None,:],valid.any(-1)[:,None]

    def forward(self,x,length):
        values,present=self.encode(x,length)
        assert values.shape[1]==1,'Use encode for retained multiple tokens'
        return values[:,0]


class RecoveryModel(nn.Module):
    def __init__(self,family='cnn',local_pool='mean',aggregation_stage=None,
                 aggregation_method='mil',task_indices=None,d_model=32,dropout=.1):
        super().__init__()
        self.config=dict(family=family,local_pool=local_pool,aggregation_stage=aggregation_stage,
            aggregation_method=aggregation_method,task_indices=task_indices,d_model=d_model,dropout=dropout)
        self.family=family;self.local_pool=local_pool;self.aggregation_stage=aggregation_stage
        self.aggregation_method=aggregation_method;self.task_indices=task_indices
        self.base=RawTemporalModel(family,d_model,dropout)
        self.base.local=LocalReadout(self.base.local,local_pool,d_model)
        self.stage_projection=None
        if aggregation_method=='mean_max':
            self.stage_projection=nn.Linear(2*d_model,d_model)
            nn.init.zeros_(self.stage_projection.weight);nn.init.zeros_(self.stage_projection.bias)
            with torch.no_grad():self.stage_projection.weight[:,:d_model].copy_(torch.eye(d_model))

    @property
    def local(self):return self.base.local

    def aggregate(self,tokens,mask,stage,duration=None):
        method=self.aggregation_method if stage==self.aggregation_stage else 'mil'
        count=mask.sum(1).to(tokens.dtype)
        if method=='mil':
            pooled,attention=self.base.pool(tokens,mask);return pooled,count,attention
        if method=='duration':
            assert duration is not None
            weights=duration.clamp_min(1e-4)*mask;mass=weights.sum(1)
        else:
            weights=mask.to(tokens.dtype);mass=count
        weights=weights/mass.clamp_min(1e-8)[:,None]
        mean=(tokens*weights[:,:,None]).sum(1)
        if method=='mean_max':
            maximum=tokens.masked_fill(~mask[:,:,None],-torch.inf).amax(1)
            maximum=torch.where(mask.any(1)[:,None],maximum,0.)
            mean=self.stage_projection(torch.cat([mean,maximum],-1))
        else:assert method in ('mean','duration')
        return mean,mass,weights

    def grouped(self,values,present,indices,stage,context=None,duration=None):
        safe=indices.clamp_min(0);mask=(indices>=0)&present[safe]
        tokens=values[safe].masked_fill(~mask[:,:,None],0.)
        if isinstance(context,TemporalBlock):tokens,_=context(tokens,mask)
        elif context is not None:tokens=context(tokens,mask)
        lengths=duration[safe] if duration is not None else None
        return self.aggregate(tokens,mask,stage,lengths)

    def forward(self,batch,patch_present=None,task_present=None,return_details=False):
        base=self.base;device=batch['x'].device;number=batch.get('person_count',1);tasks=14*number
        taskmask=torch.ones(tasks,device=device,dtype=torch.bool) if task_present is None else task_present.clone()
        if self.task_indices is not None:
            allowed=torch.zeros(14,device=device,dtype=torch.bool);allowed[self.task_indices]=True
            taskmask&=allowed.repeat(number)
        tokens,valid=self.local.encode(batch['x'],batch['length'])
        patches,retained,dim=tokens.shape
        active=torch.ones(patches,device=device,dtype=torch.bool) if patch_present is None else patch_present
        patchtask=batch['patch_task'].repeat_interleave(retained)
        patchsegment=batch['patch_segment'].repeat_interleave(retained)
        present=(valid&active[:,None]).reshape(-1)&taskmask[patchtask]
        z=base.drop(tokens.reshape(-1,dim));details={'window_tokens':tokens,'window_token_valid':valid}
        if retained==1:
            metadata=batch['patch_meta'];stroke_groups=batch['stroke_groups'];flat_groups=batch['flat_groups']
            duration=batch.get('patch_duration')
        else:
            assert retained==4 and batch['x'].shape[-1]==128
            metadata=batch['token_meta'].reshape(-1,4)
            stroke_groups=torch.as_tensor(groups(patchsegment.detach().cpu().numpy(),16*retained),device=device)
            flat_groups=torch.as_tensor(groups(patchtask.detach().cpu().numpy(),32*retained),device=device)
            duration=torch.expm1(metadata[:,1]*4)
        if self.family not in ('no_explicit_time','no_position'):z=z+base.time(metadata)*present[:,None]
        if base.hier:
            chunks,mass,attention=self.grouped(z,present,stroke_groups,'window_to_stroke',base.stroke,duration)
            destination=patchsegment[stroke_groups[:,0]]
            segment,segment_present=scatter_average(chunks,destination,mass,len(batch['segment_task']))
            details.update(segment=segment,segment_present=segment_present,stroke_pool=attention)
            if self.family not in ('no_explicit_time','no_position'):segment=segment+base.time(batch['segment_meta'])
            allowed=segment_present&taskmask[batch['segment_task']]
            chunks,mass,attention=self.grouped(segment,allowed,batch['task_groups'],'stroke_to_task',base.task_context,batch.get('segment_duration'))
            task,validtask=scatter_average(chunks,batch['task_group_task'],mass,tasks)
            details['task_pool']=attention
        else:
            context=base.task_context if base.task_context is not None else base.tcn
            chunks,mass,attention=self.grouped(z,present,flat_groups,'window_to_stroke',context,duration)
            task,validtask=scatter_average(chunks,patchtask[flat_groups[:,0]],mass,tasks)
            details['flat_pool']=attention
        validtask=(validtask&taskmask).reshape(number,14)
        details['task_pre_ff']=task.reshape(number,14,dim)
        task=base.task_ff(task+base.task_id(torch.arange(tasks,device=device)%14)).reshape(number,14,dim)
        details['task_pre_context']=task
        if base.person_context is not None:task,attention=base.person_context(task,validtask);details['task_attention']=attention
        person,_,attention=self.aggregate(task,validtask,'task_to_person',batch.get('task_duration',None).reshape(number,14) if 'task_duration' in batch else None)
        details.update(task=task,task_valid=validtask,person=person,participant_pool=attention)
        logit=base.head(person).reshape(number)
        return (logit,details) if return_details else logit
