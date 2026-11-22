import os,random
import numpy as np
import torch
from .common import sha,write
def save(path,state):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)
 write(str(path.relative_to(__import__('scripts.native_channel_research.common',fromlist=['LOCAL']).LOCAL))+'.sha256.json',{'sha256':sha(path)},local=True)
def load(path):
 import json
 sig=json.loads(path.with_suffix(path.suffix+'.sha256.json').read_text())
 if sha(path)!=sig['sha256']:raise ValueError('Checkpoint hash mismatch')
 return torch.load(path,map_location='cpu',weights_only=False) # Only this pipeline local checkpoints, hash checked.
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
 if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(n)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
