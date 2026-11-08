"""Re-extract every original task read-only to check image/cache comparability."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))
import time
import numpy as np
import torch
from embedding_common import OUT,BRANCHES,ENCODERS,initialize,sha,write_json
from extract import load_encoder
from extract_embedding_ablation import images_for


def main():
    initialize(); torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    signals=np.load(ROOT/'data/derived/signals512.npy',mmap_mode='r').reshape(-1,7,512)
    rows=[]
    for branch,key in BRANCHES.items():
        model,dim=load_encoder(ENCODERS[branch]); model.eval().requires_grad_(False).cuda()
        reference=np.load(ROOT/'data/derived'/(key+'.npz'))['X'].reshape(-1,dim)
        maximum=0.; absolute_sum=0.; count=0; start=time.perf_counter()
        with torch.inference_mode():
            for lo in range(0,len(signals),16):
                images=images_for(np.array(signals[lo:lo+16]),branch,lo)
                # Match original NumPy normalization exactly, before device transfer.
                values=model(torch.from_numpy((images-.449)/.226).cuda()).cpu().numpy()
                old=reference[lo:lo+len(values)]
                np.testing.assert_allclose(values,old,atol=3e-4,rtol=3e-4)
                diff=np.abs(values-old)
                maximum=max(maximum,float(diff.max())); absolute_sum+=float(diff.sum()); count+=diff.size
        rows.append(dict(branch=branch,tasks_checked=len(signals),allclose_passed=True,
                         atol=3e-4,rtol=3e-4,max_absolute_error=maximum,
                         mean_absolute_error=absolute_sum/count,seconds=time.perf_counter()-start))
        print('BASELINE AUDIT',branch,len(signals),'tasks, max abs error',maximum,flush=True)
        del model; torch.cuda.empty_cache()
    write_json(OUT/'baseline_extraction_audit.json',dict(settings_sha256=sha(OUT/'settings.json'),
        all_tasks_verified=True,original_embeddings_overwritten=False,records=rows))


if __name__=='__main__': main()
