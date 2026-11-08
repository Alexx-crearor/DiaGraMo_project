"""Matched warm GPU inference benchmark; fixed label-free first 32 tasks, no tuning."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))
import time
import statistics
import numpy as np
import torch
from embedding_common import OUT,BRANCHES,ENCODERS,MODES,initialize,sha,write_json
from extract import load_encoder
from extract_embedding_ablation import images_for
from diagramo.channel_embedding import FixedChannelAdapter,channel_rgb,aggregate_channel_embeddings


def main():
    initialize(); torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    signals=np.load(ROOT/'data/derived/signals512.npy',mmap_mode='r').reshape(-1,7,512)
    records=[]
    for branch in BRANCHES:
        t=time.perf_counter(); images=images_for(np.array(signals[:32]),branch,0)
        image_seconds=time.perf_counter()-t
        rgb,dim=load_encoder(ENCODERS[branch],3); rgb.eval().requires_grad_(False).cuda()
        original,_=load_encoder(ENCODERS[branch]); original.eval().requires_grad_(False).cuda()
        adapter=FixedChannelAdapter().cuda()
        def run(mode):
            values=[]
            for lo in range(0,len(images),16):
                x=torch.from_numpy(images[lo:lo+16]).cuda()
                if mode=='original': result=original((x-.449)/.226).cpu()
                elif mode=='group_adapter': result=rgb((adapter(x)-.449)/.226).cpu()
                else:
                    single=channel_rgb(x)
                    pieces=[rgb((single[i:i+16]-.449)/.226).cpu() for i in range(0,len(single),16)]
                    concat,mean=aggregate_channel_embeddings(torch.cat(pieces),len(x))
                    result=concat if mode=='channel_concat' else mean
                values.append(result)
            return torch.cat(values)
        with torch.inference_mode():
            for mode in MODES:
                run(mode); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                times=[]
                for _ in range(3):
                    t=time.perf_counter(); values=run(mode); torch.cuda.synchronize()
                    times.append(time.perf_counter()-t)
                records.append(dict(branch=branch,mode=mode,n_tasks=32,repeats=3,batch_size=16,
                    seconds=times,median_seconds=statistics.median(times),
                    min_seconds=min(times),max_seconds=max(times),
                    estimated_ms_per_participant=1000*statistics.median(times)/32*14,
                    peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                    raw_features=14*dim*(7 if mode=='channel_concat' else 1),
                    image_generation_seconds=image_seconds))
                print('BENCHMARK',branch,mode,round(statistics.median(times),3),'s/32 tasks',flush=True)
        del rgb,original; torch.cuda.empty_cache()
    write_json(OUT/'benchmark.json',dict(settings_sha256=sha(OUT/'settings.json'),
        scope='warm extraction including host/device transfers and CPU aggregation; excludes image generation, model loading, reducer and classifier',
        label_access=False,device=torch.cuda.get_device_name(),records=records))


if __name__=='__main__': main()
