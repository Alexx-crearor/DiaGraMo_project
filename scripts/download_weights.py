"""Download public pretrained weights to project-local caches; no data uploaded."""
import os,json,hashlib,urllib.request,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
URLS=['resnet50-11ad3fa6.pth','efficientnet_b4_rwightman-23ab8bcd.pth','vit_b_16-c867db91.pth','vgg16-397923af.pth']

def fetch(name):
    folder=ROOT/'.cache/torch/hub/checkpoints'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/name; prefix=name.rsplit('-',1)[1].split('.')[0]
    if not path.exists():
        part=path.with_suffix('.download')
        urllib.request.urlretrieve('https://download.pytorch.org/models/'+name,part)
        digest=hashlib.sha256(part.read_bytes()).hexdigest()
        if not digest.startswith(prefix): raise RuntimeError('Hash mismatch '+name)
        part.replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(prefix): raise RuntimeError('Cached hash mismatch '+name)
    print('Downloaded and verified',name,flush=True)
    return dict(name=name,sha256=digest,bytes=path.stat().st_size)

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(fetch,URLS))
    from huggingface_hub import snapshot_download,HfApi
    model='AutonLab/MOMENT-1-large'; info=HfApi().model_info(model)
    path=snapshot_download(model,revision=info.sha,allow_patterns=['config.json','model.safetensors','pytorch_model.bin'])
    rows.append(dict(name=model,revision=info.sha,path=path))
    target=ROOT/'references/weights_manifest.json'; target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(rows,indent=2),encoding='utf-8'); print('COMPLETE',target,flush=True)

if __name__=='__main__': main()
