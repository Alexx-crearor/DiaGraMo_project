"""Frozen pretrained encoders, with resumable per-task feature extraction."""
import os,sys,json,argparse,time,gc
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
import numpy as np
import torch
from diagramo.features import resample,gaf,cwt

def save_json(p,obj): p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def load_encoder(name, input_channels=7):
    """Load the original seven-channel baseline, or an unchanged RGB encoder.

    input_channels=7 preserves the original implementation and output paths.
    RGB is used only by the separate embedding ablation extraction script.
    """
    from torchvision import models
    from torch import nn
    if name=='resnet50':
        model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc=nn.Identity(); first='conv1'; dim=2048
    elif name=='efficientnet_b4':
        model=models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier=nn.Identity(); first='features.0.0'; dim=1792
    elif name=='vit_b_16':
        model=models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads=nn.Identity(); first='conv_proj'; dim=768
    elif name=='vgg16':
        model=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[-1]=nn.Identity(); first='features.0'; dim=4096
    else: raise ValueError(name)
    if input_channels==3: return model,dim
    if input_channels!=7: raise ValueError('Supported encoder inputs: 3 or 7 channels')
    parent=model; parts=first.split('.')
    for part in parts[:-1]: parent=getattr(parent,part)
    old=getattr(parent,parts[-1])
    new=nn.Conv2d(7,old.out_channels,old.kernel_size,old.stride,old.padding,bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight.sum(1,keepdim=True).repeat(1,7,1,1)/7)
        if old.bias is not None: new.bias.copy_(old.bias)
    setattr(parent,parts[-1],new)
    return model,dim

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--representation',choices=['moment','gaf','cwt'],required=True)
    ap.add_argument('--encoder',choices=['resnet50','efficientnet_b4','vit_b_16','vgg16'],default='vit_b_16')
    ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--device',default='auto'); ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--precision',choices=['float32','float16'],default='float32')
    args=ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(42)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    signals=np.load(ROOT/'data/derived/signals512.npy',mmap_mode='r')
    labels=np.load(ROOT/'data/derived/kinematic.npz'); n,nt=signals.shape[:2]
    name='moment' if args.representation=='moment' else args.representation+'_'+args.encoder
    out=ROOT/'data/derived'; state=out/(name+'.progress.json'); target=out/(name+'.tasks.npy'); final=out/(name+'.npz')
    if final.exists(): print(f'{name} already complete',flush=True); return
    previous=json.loads(state.read_text()) if state.exists() and target.exists() else {}
    start=previous.get('completed_tasks',0)
    if previous and previous.get('precision','float16')!=args.precision:
        print('Precision changed: recomputing the partial cache consistently from task 0',flush=True); start=0
    if args.representation=='moment':
        from momentfm import MOMENTPipeline
        manifest=ROOT/'references/weights_manifest.json'
        revision=next((r.get('revision') for r in json.loads(manifest.read_text()) if r['name']=='AutonLab/MOMENT-1-large'),None) if manifest.exists() else None
        model=MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-large',revision=revision,model_kwargs={'task_name':'embedding'})
        model.init(); dim=1024
    else: model,dim=load_encoder(args.encoder)
    model.eval().requires_grad_(False).to(device)
    # On Quadro T1000 float32 is materially faster (see measured benchmark).
    half=device=='cuda' and args.precision=='float16'
    if half: model.half()
    result=np.lib.format.open_memmap(target,mode='r+' if start else 'w+',dtype='float32',shape=(n*nt,dim))
    flat=signals.reshape(n*nt,7,512); started=time.time()
    print(f'{name}: device={device}, subjects={n}, tasks={n*nt}, start={start}, dim={dim}',flush=True)
    for i in range(start,n*nt,args.batch_size):
        s=np.array(flat[i:i+args.batch_size],dtype=np.float32)
        if args.representation=='gaf':
            z=np.stack([np.stack([resample(a,224) for a in task]) for task in s])
            images=(gaf(z)+1)/2
        elif args.representation=='cwt':
            # Cache separately per task to avoid repeating costly wavelets across 4 encoders.
            cache=ROOT/'data/derived/cwt_cache'; cache.mkdir(exist_ok=True)
            ims=[]
            for j,task in enumerate(s):
                p=cache/f'{i+j:05d}.npy'
                if p.exists(): im=np.load(p).astype('float32')
                else:
                    im=cwt(task); temp=p.with_suffix('.extract.tmp')
                    with temp.open('wb') as f: np.save(f,im.astype('float16'))
                    temp.replace(p)
                ims.append(im)
            images=np.stack(ims)
        with torch.inference_mode():
            if args.representation=='moment':
                x=torch.from_numpy(s).to(device)
                # Keep raw large-range signals float32 through RevIN; half variance
                # would overflow for pressure. Autocast converts linear layers only.
                context=torch.autocast('cuda',dtype=torch.float16) if half else nullcontext()
                with context: emb=model(x_enc=x).embeddings
            else:
                x=torch.from_numpy((images-.449)/.226).to(device)
                if half: x=x.half()
                emb=model(x)
        a=emb.float().cpu().numpy()
        if a.shape!=(len(s),dim) or not np.isfinite(a).all(): raise ValueError(f'Invalid embeddings {name} at {i}')
        result[i:i+len(s)]=a; result.flush()
        completed=i+len(s); save_json(state,dict(completed_tasks=completed,total_tasks=n*nt,device=device,precision=args.precision))
        if i%max(args.batch_size,100//args.batch_size*args.batch_size)==0:
            elapsed=time.time()-started; rate=(completed-start)/max(elapsed,.01)
            print(f'{name}: {completed}/{n*nt}, {rate:.2f} tasks/s, ETA {(n*nt-completed)/max(rate,.01)/60:.1f} min',flush=True)
    np.savez_compressed(final,X=np.asarray(result).reshape(n,nt*dim),y=labels['y'],ids=labels['ids'])
    save_json(out/(name+'.metadata.json'),dict(representation=args.representation,encoder='AutonLab/MOMENT-1-large' if name=='moment' else args.encoder,
        dimension_per_task=dim,dimension_per_subject=dim*nt,subjects=n,frozen=True,device=device,
        precision=args.precision,channel_adaptation='sum RGB kernels / 7, repeated 7 channels',
        image_normalization='(image[0,1] - 0.449)/0.226',seconds=time.time()-started))
    print(f'COMPLETE {final}',flush=True)

if __name__=='__main__': main()
