"""Extract all label-free RGB adaptations without overwriting original embeddings."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))
import argparse
import time
import numpy as np
import torch
from embedding_common import OUT,BRANCHES,ENCODERS,sha,write_json,initialize
from extract import load_encoder
from diagramo.features import gaf,cwt,resample,CHANNELS
from diagramo.channel_embedding import FixedChannelAdapter,channel_rgb,aggregate_channel_embeddings


def images_for(signals,branch,start):
    if branch=='GAF':
        z=np.stack([np.stack([resample(a,224) for a in task]) for task in signals])
        return (gaf(z)+1)/2
    images=[]
    for j,task in enumerate(signals):
        p=ROOT/'data/derived/cwt_cache'/f'{start+j:05d}.npy'
        # Old cache is read-only; a missing entry is recomputed in memory only.
        images.append(np.load(p).astype('float32') if p.exists() else cwt(task).astype('float16').astype('float32'))
    return np.stack(images)


def extract(branch,batch_size):
    config=initialize(); signature=sha(OUT/'settings.json')
    dest=OUT/'embeddings'; dest.mkdir(exist_ok=True)
    metadata=dest/f'{branch}.json'
    if metadata.exists():
        import json
        old=json.loads(metadata.read_text())
        if old['settings_sha256']!=signature: raise ValueError('Extraction configuration differs')
        if all(sha(dest/v['file'])==v['sha256'] for v in old['outputs'].values()):
            print('REUSE',branch,'completed extraction',flush=True); return
        raise ValueError('Extraction file integrity mismatch')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this measured T1000 study')
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model,dim=load_encoder(ENCODERS[branch],input_channels=3)
    model.eval().requires_grad_(False).cuda()
    adapter=FixedChannelAdapter().cuda()
    signals=np.load(ROOT/'data/derived/signals512.npy',mmap_mode='r')
    n,nt=signals.shape[:2]; flat=signals.reshape(-1,7,512)
    sizes={'group_adapter':dim,'channel_mean':dim,'channel_concat':7*dim}
    progress=dest/f'{branch}.progress.json'
    import json
    state=json.loads(progress.read_text()) if progress.exists() else {}
    if state and state['settings_sha256']!=signature: raise ValueError('Partial extraction settings mismatch')
    start=state.get('completed_tasks',0)
    buffers={m:np.lib.format.open_memmap(dest/f'{branch}__{m}.tasks.npy',
             mode='r+' if start else 'w+',dtype='float32',shape=(len(flat),d)) for m,d in sizes.items()}
    costs=state.get('costs',dict(images_seconds=0.,group_adapter_seconds=0.,channel_separate_seconds=0.))
    control={}
    with torch.inference_mode():
        # Recompute original and mathematically equivalent mean-RGB on the same images.
        ims=images_for(np.array(flat[:2]),branch,0)
        x=torch.from_numpy(ims).cuda()
        original,_=load_encoder(ENCODERS[branch])
        original.eval().requires_grad_(False).cuda()
        a=original((x-.449)/.226).cpu().numpy()
        b=model(((x.mean(1,keepdim=True).repeat(1,3,1,1))-.449)/.226).cpu().numpy()
        np.testing.assert_allclose(a,b,atol=3e-4,rtol=3e-4)
        baseline=np.load(ROOT/'data/derived'/(BRANCHES[branch]+'.npz'))['X'].reshape(n*nt,dim)[:2]
        np.testing.assert_allclose(a,baseline,atol=3e-4,rtol=3e-4)
        control=dict(mean_rgb_max_abs_error=float(np.max(np.abs(a-b))),
                     old_embedding_max_abs_error=float(np.max(np.abs(a-baseline))),n_tasks=2)
        del original
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        wall=time.perf_counter()
        for i in range(start,len(flat),batch_size):
            began=time.perf_counter()
            ims=images_for(np.array(flat[i:i+batch_size]),branch,i)
            x=torch.from_numpy(ims).cuda()
            torch.cuda.synchronize(); costs['images_seconds']+=time.perf_counter()-began
            began=time.perf_counter()
            group=model((adapter(x)-.449)/.226)
            torch.cuda.synchronize(); costs['group_adapter_seconds']+=time.perf_counter()-began
            began=time.perf_counter()
            single=channel_rgb(x); pieces=[]
            for lo in range(0,len(single),batch_size):
                pieces.append(model((single[lo:lo+batch_size]-.449)/.226).cpu())
            concat,mean=aggregate_channel_embeddings(torch.cat(pieces),len(x))
            torch.cuda.synchronize(); costs['channel_separate_seconds']+=time.perf_counter()-began
            values={'group_adapter':group.cpu().numpy(),'channel_concat':concat.numpy(),'channel_mean':mean.numpy()}
            for mode,value in values.items():
                if not np.isfinite(value).all(): raise ValueError('Invalid embedding')
                buffers[mode][i:i+len(x)]=value
                buffers[mode].flush()
            done=i+len(x)
            write_json(progress,dict(settings_sha256=signature,completed_tasks=done,total_tasks=len(flat),costs=costs))
            if i%max(batch_size,128//batch_size*batch_size)==0:
                rate=(done-start)/max(time.perf_counter()-wall,.01)
                print(f'{branch}: {done}/{len(flat)} tasks | {rate:.1f} tasks/s | ETA {(len(flat)-done)/max(rate,.01):.0f}s',flush=True)
    outputs={}
    for mode,d in sizes.items():
        path=dest/f'{branch}__{mode}.npy'
        np.save(path,np.asarray(buffers[mode]).reshape(n,nt*d))
        outputs[mode]=dict(file=path.name,sha256=sha(path),shape=[n,nt*d],encoder_images_per_task=1 if mode=='group_adapter' else 7)
    write_json(metadata,dict(settings_sha256=signature,branch=branch,encoder=ENCODERS[branch],
               participants=n,tasks_per_participant=nt,channel_order=CHANNELS,
               dtype='float32',device=torch.cuda.get_device_name(),batch_size=batch_size,
               frozen=True,label_access=False,fit_participants=[],adapter_weights=adapter.weight.cpu().flatten(1).tolist(),
               equivalence_control=control,costs=costs,peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
               outputs=outputs))
    print('COMPLETE EXTRACTION',branch,costs,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',choices=list(BRANCHES),required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    extract(args.branch,args.batch_size)
