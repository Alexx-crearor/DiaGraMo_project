import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pytest
import torch
from scripts.native_channel_research.participant_dataset import fit_scaler,windows
from scripts.native_channel_research.sequence_permutation import permute_rows
from scripts.native_channel_research.compact_multiscale_cnn import CompactCNN
from scripts.native_channel_research.resume import save,load,seed,rng_state,restore
from scripts.native_channel_research.opaque_feature_extractor import features,length_features
from scripts.native_channel_research.train import tensors
from scripts.native_channel_research.common import ROOT,LOCAL,sha

def test_scaler_equal_participants_train_only():
 d=[[np.ones((1000,7))],[np.ones((2,7))*3],[np.ones((4,7))*999]]
 m,s=fit_scaler(d,[0,1]);assert np.array_equal(m,np.ones(7)*2);assert np.array_equal(s,np.ones(7))
 d[2][0]*=100;assert np.array_equal(fit_scaler(d,[0,1])[0],m)
@pytest.mark.parametrize('n',[1,255,256,257,512,513])
def test_window_coverage(n):
 x=np.arange(n*7).reshape(n,7);a,m,g,w=windows([x],np.zeros(7),np.ones(7))
 assert int(m.sum())==n and int(w.sum())==n
 assert np.array_equal(a.transpose(0,2,1)[m.astype(bool)],x)
def test_recording_boundaries_and_terminal():
 a,m,g,w=windows([np.ones((257,7)),np.ones((3,7))*2],np.zeros(7),np.ones(7))
 assert list(g)==[0,0,1] and list(w)==[256,1,3]
 assert np.all(a[2,:,:3]==2)
def test_permutation_multiset_joint_rows_seed():
 x=np.arange(256*7).reshape(256,7);y=permute_rows(x,'a')
 assert not np.array_equal(x,y);assert np.array_equal(y,permute_rows(x,'a'))
 assert sorted(map(tuple,x))==sorted(map(tuple,y))
def test_permutation_padding():
 x=np.arange(13*7).reshape(13,7);a,m,g,w=windows([x],np.zeros(7),np.ones(7),True,'a')
 assert not a[:,:,13:].any();assert sorted(map(tuple,a[0,:,:13].T))==sorted(map(tuple,x))
def test_model_match_and_aggregation():
 seed(42);a=CompactCNN();seed(42);b=CompactCNN()
 assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
 assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
 a.eval();w=windows([np.ones((257,7))],np.zeros(7),np.ones(7));t=tensors(w,'cpu')
 assert torch.allclose(a(*t,chunk=1),a(*t,chunk=64),atol=1e-6)
def test_fixed_seed_gradients():
 for _ in range(2):
  seed(42);m=CompactCNN();w=tensors(windows([np.ones((31,7))],np.zeros(7),np.ones(7)),'cpu');v=m(*w);v.backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters())
  current=[p.detach().clone() for p in m.parameters()]
  if _==0:first=current
  else:assert all(torch.equal(a,b) for a,b in zip(first,current))
def test_optimizer_rng_resume_exact():
 seed(42);m=CompactCNN();o=torch.optim.AdamW(m.parameters(),lr=.001)
 w=tensors(windows([np.arange(217).reshape(31,7)],np.zeros(7),np.ones(7)),'cpu')
 def step():
  o.zero_grad();m(*w).square().backward();o.step()
 step();state={k:v.clone() for k,v in m.state_dict().items()};import copy
 opt=copy.deepcopy(o.state_dict());rng=rng_state();step();expected={k:v.clone() for k,v in m.state_dict().items()}
 m.load_state_dict(state);o.load_state_dict(opt);restore(rng);step()
 assert all(torch.equal(v,expected[k]) for k,v in m.state_dict().items())
def test_checkpoint_integrity_and_resume():
 path=LOCAL/'test_checkpoint.pt';seed(42);save(path,dict(value=torch.arange(3),rng=rng_state()))
 assert torch.equal(load(path)['value'],torch.arange(3))
 raw=path.read_bytes();path.write_bytes(raw+b'bad')
 with pytest.raises(ValueError):load(path)
 path.write_bytes(raw);assert torch.equal(load(path)['value'],torch.arange(3))
def test_feature_contract():
 x=np.arange(217).reshape(31,7);assert features(x).shape==(105,);assert length_features([x]).shape==(6,)
def test_actual_grouping_source_and_folds():
 from scripts.native_channel_research.participant_dataset import load_data
 p,d=load_data()
 for f in range(5):
  tr=set(p[p.candidate_fold!=f].participant_id);te=set(p[p.candidate_fold==f].participant_id)
  assert not tr&te and len(tr|te)==120
 assert len(d)==120

def test_private_git_ignored():
 import subprocess
 assert subprocess.run(['git','check-ignore','data/native_channel_research_local/private.csv'],capture_output=True).returncode==0
 assert not subprocess.check_output(['git','ls-files','data/native_channel_research_local'],text=True).strip()

def test_complete_oof_gate():
 from scripts.native_channel_research.evaluate import complete_oof
 import pandas as pd
 with pytest.raises(ValueError):complete_oof(pd.DataFrame({'participant_id':['x'],'fold':[0]}))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable; report limitation')
def test_cuda_cpu_and_exact_cuda_resume():
 seed(42);torch.set_num_threads(4);gpu=CompactCNN().cuda();cpu=CompactCNN();cpu.load_state_dict({k:v.cpu() for k,v in gpu.state_dict().items()});gpu.eval();cpu.eval()
 a=np.random.default_rng(42).normal(size=(257,7));w=windows([a],np.zeros(7),np.ones(7))
 with torch.no_grad():assert torch.allclose(gpu(*tensors(w,'cuda')).cpu(),cpu(*tensors(w,'cpu')),atol=1e-4,rtol=1e-4)
 gpu.train();opt=torch.optim.AdamW(gpu.parameters(),lr=.001);inp=tensors(w,'cuda')
 def step():
  opt.zero_grad();gpu(*inp).square().backward();assert all(torch.isfinite(p.grad).all() for p in gpu.parameters());opt.step()
 import copy
 step();path=LOCAL/'test_cuda_exact_resume.pt'
 save(path,dict(model=gpu.state_dict(),optimizer=opt.state_dict(),rng=rng_state()))
 step();expected={k:v.clone() for k,v in gpu.state_dict().items()}
 saved=load(path);gpu.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
 for values in opt.state.values():
  for key,value in values.items():
   if torch.is_tensor(value):values[key]=value.cuda()
 restore(saved['rng']);step()
 assert all(torch.equal(v,expected[k]) for k,v in gpu.state_dict().items())

def test_label_free_preprocessing_api():
 import inspect
 assert 'label' not in inspect.signature(fit_scaler).parameters
 assert 'label' not in inspect.signature(windows).parameters
 assert 'label' not in inspect.signature(features).parameters

def test_protected_inputs_unchanged():
 from scripts.native_channel_research.common import read
 for row in read('PROTECTED_INPUT_MANIFEST.json')['files']:assert sha(ROOT/row['path'])==row['sha256']

def test_participant_equal_recording_pool():
 seed(42);m=CompactCNN();m.eval()
 w=windows([np.ones((257,7)),np.ones((3,7))*2],np.zeros(7),np.ones(7));x,mask,groups,weights=tensors(w,'cpu')
 with torch.no_grad():
  h=m.window_encode(x,mask);r0=(h[:2]*weights[:2,None]).sum(0)/weights[:2].sum();r1=h[2];expected=m.head(torch.relu(m.project((r0+r1)/2))).squeeze()
  assert torch.allclose(m(x,mask,groups,weights),expected,atol=1e-6)

def test_single_gpu_worker_lock(tmp_path):
 from scripts.native_channel_research.common import worker_lock
 with worker_lock(tmp_path/'worker.lock'):
  with pytest.raises((OSError,BlockingIOError)):
   with worker_lock(tmp_path/'worker.lock'):pass
