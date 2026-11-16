"""Participant-level ragged batches; normalization learns only supplied Train IDs."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .preprocessing import CONTINUOUS
from .segmentation import groups

@lru_cache(maxsize=1)
def memory_bank(cache_folder):
    import json
    metadata=json.loads((Path(cache_folder)/'packed.json').read_text())
    return np.load(Path(cache_folder)/'packed.npy',mmap_mode='c'),metadata['offsets']

def time_metadata(inventory):
    a=inventory;start=a[:,5]-a[0,5];duration=a[:,6]-a[:,5];count=a[:,1]-a[:,0]
    values=np.column_stack([start,duration,np.divide(duration,np.maximum(count-1,1)),count])
    return (np.sign(values)*np.log1p(np.abs(values))/4).astype('float32')

@lru_cache(maxsize=258)
def load_person(cache_folder,pid):
    packed=(Path(cache_folder)/'packed.json').exists()
    with np.load(Path(cache_folder)/(pid+'.npz'))as a:
        pieces=[];indexpieces=[];length=[];task=[];segment=[];metadata=[];segment_task=[];segment_metadata=[];segment_weights=[];inventories=[]
        offset=0
        for ti in range(14):
            inv=a[f'm{ti}'];n=a[f'n{ti}'];length.append(n)
            if not packed:pieces.append(a[f'x{ti}']);indexpieces.append(a[f'i{ti}'])
            task.append(np.full(len(n),ti,dtype='int64'));local=inv[:,2].astype('int64');segment.append(local+offset)
            metadata.append(time_metadata(inv));inventories.append(inv)
            for s in range(local.max()+1):
                m=np.flatnonzero(local==s);row=inv[m[0]].copy();row[1]=inv[m[-1],1];row[6]=inv[m[-1],6]
                segment_task.append(ti);segment_metadata.append(row);segment_weights.append(n[m].sum())
            offset+=local.max()+1
    # Compact RAM storage only: math/gradients remain float32. Original float32
    # caches and float64 timestamp/index inventories remain untouched on disk.
    if packed:
        bank,offsets=memory_bank(cache_folder);a,b=offsets[pid];x=bank[0,a:b];ix=bank[1,a:b]
    else:x=np.concatenate(pieces).astype('float16');ix=np.concatenate(indexpieces).astype('float16')
    n=np.concatenate(length);pt=np.concatenate(task);ps=np.concatenate(segment)
    sm=np.asarray(segment_metadata);st=np.asarray(segment_task,dtype='int64')
    # Segment time metadata uses its own task reference, matching original task clocks.
    sfeatures=np.zeros((len(sm),4),dtype='float32')
    for ti in range(14):sfeatures[st==ti]=time_metadata(sm[st==ti])
    sg=groups(ps,16);tg=groups(st,32);fg=groups(pt,32)
    return dict(x=x,index_x=ix,length=n,patch_task=pt,patch_segment=ps,patch_meta=np.concatenate(metadata),segment_meta=sfeatures,
       segment_task=st,segment_weight=np.asarray(segment_weights,dtype='float32'),stroke_groups=sg,stroke_group_segment=ps[sg[:,0]],
       task_groups=tg,task_group_task=st[tg[:,0]],flat_groups=fg,flat_group_task=pt[fg[:,0]],inventories=inventories)

def sufficient_statistics(person):
    result=[]
    mask=np.arange(person['x'].shape[1])[None,:]<person['length'][:,None]
    for key in ('x','index_x'):
        A=person[key][mask].astype('float64');result.extend([A.mean(0),np.square(A).mean(0)])
    return np.asarray(result)

class TrainNormalizer:
    def fit(self,statistics,train_indices,train_ids,index_only=False):
        a=statistics[np.asarray(train_indices)];offset=2 if index_only else 0
        self.mean=a[:,offset].mean(0);second=a[:,offset+1].mean(0)
        self.scale=np.maximum(np.sqrt(np.maximum(second-self.mean**2,0.)),.1)
        flags=np.setdiff1d(np.arange(len(self.mean)),CONTINUOUS);self.mean[flags]=0.;self.scale[flags]=1.
        self.fit_ids=list(train_ids);self.index_only=index_only
        return self
    def transform(self,person,dev='cuda',family='hierarchical'):
        x=person['index_x']if self.index_only else person['x']
        mask=np.arange(x.shape[1])[None,:]<person['length'][:,None]
        z=torch.as_tensor(x,device=dev).float()
        z=((z-torch.tensor(self.mean,device=dev,dtype=torch.float32))/torch.tensor(self.scale,device=dev,dtype=torch.float32)).clamp(-10,10)
        valid=torch.as_tensor(mask,device=dev);z=z.masked_fill(~valid[:,:,None],0.)
        if family=='no_explicit_time':z[:,:,5:7]=0.;z[:,:,15:20]=0.
        tensors={k:torch.as_tensor(v,device=dev)for k,v in person.items()if k not in ('x','index_x','inventories')}
        tensors['x']=z.transpose(1,2).contiguous()
        tensors['sample_mask']=torch.as_tensor(mask,device=dev)
        if family=='index_time':
            for meta,name in [('patch_meta','patch_task'),('segment_meta','segment_task')]:
                # Sample-index encoding contains no clock-valued coordinate.
                value=torch.zeros_like(tensors[meta]);idx=tensors[name];
                for ti in range(14):
                    at=torch.nonzero(idx==ti).flatten();value[at,0]=torch.log1p(torch.arange(len(at),device=dev,dtype=torch.float32))/4
                tensors[meta]=value
        return tensors

def collate_people(items):
    """Join ragged windows while keeping every hierarchy/participant boundary."""
    result={};offset_p=offset_s=offset_t=0;values={}
    for item in items:
        for key,v in item.items():
            a=v
            if key in ('patch_task','segment_task','task_group_task','flat_group_task'):a=v+offset_t
            elif key in ('patch_segment','stroke_group_segment'):a=v+offset_s
            elif key in ('stroke_groups','flat_groups'):a=torch.where(v>=0,v+offset_p,v)
            elif key=='task_groups':a=torch.where(v>=0,v+offset_s,v)
            values.setdefault(key,[]).append(a)
        offset_p+=len(item['length']);offset_s+=len(item['segment_task']);offset_t+=14
    for key,value in values.items():result[key]=torch.cat(value)
    result['person_count']=len(items)
    return result

