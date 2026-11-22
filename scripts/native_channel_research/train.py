import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time,json,hashlib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from .common import *
from .resume import seed,save,load,rng_state,restore
from .participant_dataset import fit_scaler,windows
from .compact_multiscale_cnn import CompactCNN
from .opaque_feature_extractor import length_features,participant_features

def tensors(w,device):return tuple(torch.as_tensor(a,device=device) for a in w)
def signature(cfg,fold,tr,te):return hashlib.sha256(json.dumps(dict(config=cfg,fold=int(fold),train=list(map(int,tr)),test=list(map(int,te))),sort_keys=True).encode()).hexdigest()
def cpu_baseline(name,p,data,cfg,budget):
 import joblib
 X=np.stack([(length_features if name=='B1' else participant_features)(r) for r in data]);out=[]
 for fold in range(5):
  budget.admit(20);tr=np.flatnonzero(p.candidate_fold!=fold);te=np.flatnonzero(p.candidate_fold==fold);sig=signature(cfg,fold,tr,te)
  path=LOCAL/f'checkpoints/{name}_fold{fold}.joblib';t=time.perf_counter()
  if path.exists():
   s=json.loads(path.with_suffix('.json').read_text());assert sha(path)==s['sha256'] and s['signature']==sig
   state=joblib.load(path);action='resumed_no_refit'
  else:
   scaler=StandardScaler().fit(X[tr]);model=LGBMClassifier(**cfg['lightgbm']);model.fit(scaler.transform(X[tr]),p.label.to_numpy()[tr]);state=(scaler,model)
   path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');joblib.dump(state,tmp);os.replace(tmp,path)
   write(str(path.relative_to(LOCAL).with_suffix('.json')),dict(sha256=sha(path),signature=sig),local=True);action='fit'
  prob=state[1].predict_proba(state[0].transform(X[te]))[:,1]
  assert np.array_equal(prob,state[1].predict_proba(state[0].transform(X[te]))[:,1])
  rows=[dict(participant_id=p.participant_id[i],label=int(p.label[i]),fold=fold,probability=float(v)) for i,v in zip(te,prob)]
  out+=rows;table(f'{name}_oof.csv',out,local=True)
  runtime_path=LOCAL/f'{name}_fold{fold}_runtime.json'
  if not runtime_path.exists():write(runtime_path.name,dict(wall_seconds=time.perf_counter()-t,action=action,gpu_training_seconds=0,checkpoint_sha256=sha(path)),local=True)
  elif action=='resumed_no_refit':write(f'{name}_fold{fold}_replay.json',dict(wall_seconds=time.perf_counter()-t,checkpoint_sha256=sha(path),no_refit=True),local=True)
 return out

def benchmark(p,data,cfg):
 device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4);seed(42)
 tr=np.flatnonzero(p.candidate_fold!=0);mean,scale=fit_scaler(data,tr);w=windows(data[tr[0]],mean,scale,key=p.participant_id[tr[0]])
 m=CompactCNN().to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
 if device=='cuda':torch.cuda.reset_peak_memory_stats()
 times=[]
 for k in range(4):
  t=time.perf_counter();opt.zero_grad();logit=m(*tensors(w,device));loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor(0.,device=device));loss.backward()
  assert all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None)
  opt.step()
  if device=='cuda':torch.cuda.synchronize()
  times.append(time.perf_counter()-t)
 # Exact CPU/GPU eval check with identical weights and input; no test metrics.
 m.eval();cpu=CompactCNN();cpu.load_state_dict({k:v.cpu() for k,v in m.state_dict().items()});cpu.eval()
 with torch.no_grad():a=float(m(*tensors(w,device)).cpu());b=float(cpu(*tensors(w,'cpu')))
 assert abs(a-b)<1e-4
 # Scale pilot by actual training window counts, not number of sample rows as IID subjects.
 windows_all=sum(sum((len(x)+255)//256 for x in rec) for rec in data)
 ratio=windows_all/len(w[0]);estimate=float(np.median(times[1:])*ratio*.8*cfg['epochs']*5*2*2)
 result=dict(device=device,participant_window_count=len(w[0]),all_windows=windows_all,measured_step_seconds=times,full_B3_B4_estimated_seconds_with_2x_margin=estimate,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0,cpu_gpu_absolute_difference=abs(a-b),finite_gradients=True,deterministic_algorithms=True,parameter_count=sum(q.numel() for q in m.parameters()),outer_test_metrics_used=False)
 write('RESOURCE_ADMISSION.json' if not (OUT/'RESOURCE_ADMISSION.json').exists() else 'RESOURCE_RECHECK.json',result);return result

def cnn(name,p,data,cfg,budget,chunk=64,force_cpu=False):
 device='cuda' if torch.cuda.is_available() and not force_cpu else 'cpu';out=[];torch.set_num_threads(4)
 for fold in range(5):
  tr=np.flatnonzero(p.candidate_fold!=fold);te=np.flatnonzero(p.candidate_fold==fold);sig=signature(cfg,fold,tr,te);seed(42+fold)
  mean,scale=fit_scaler(data,tr)
  # Cached only fold-specific transformed arrays; test is never used to fit scaler.
  prepared=[windows(rec,mean,scale,name=='B4',key=p.participant_id[i]) for i,rec in enumerate(data)]
  model=CompactCNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
  path=LOCAL/f'checkpoints/{name}_fold{fold}.pt';start_epoch=0;elapsed=0.;cpu_elapsed=0.;epoch_times=[]
  if path.exists():
   state=load(path);assert state['signature']==sig and np.array_equal(mean,state['mean']) and np.array_equal(scale,state['scale'])
   model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
   for values in opt.state.values():
    for k,v in values.items():
     if torch.is_tensor(v):values[k]=v.to(device)
   restore(state['rng']);start_epoch=state['epoch'];elapsed=state['wall_seconds'];cpu_elapsed=state['cpu_seconds'];epoch_times=state['epoch_times']
  if device=='cuda':torch.cuda.reset_peak_memory_stats()
  for epoch in range(start_epoch,cfg['epochs']):
   budget.admit(max(epoch_times,default=60)*1.5);t=time.perf_counter();ct=time.process_time();model.train();order=np.random.permutation(tr)
   for start in range(0,len(order),8):
    opt.zero_grad(set_to_none=True);batch=order[start:start+8]
    for i in batch:
     logit=model(*tensors(prepared[i],device),chunk=chunk);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor(float(p.label[i]),device=device))/len(batch)
     loss.backward()
     if device=='cuda' and torch.cuda.max_memory_allocated()>3.2*2**30:raise RuntimeError('GPU allocation budget exceeded')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
    opt.step()
   if device=='cuda':torch.cuda.synchronize()
   dt=time.perf_counter()-t;elapsed+=dt;cpu_elapsed+=time.process_time()-ct;epoch_times.append(dt)
   save(path,dict(signature=sig,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch+1,mean=mean,scale=scale,wall_seconds=elapsed,cpu_seconds=cpu_elapsed,epoch_times=epoch_times,train_indices=tr,test_indices=te,peak_allocated_bytes=max(torch.cuda.max_memory_allocated() if device=='cuda' else 0,state.get('peak_allocated_bytes',0) if path.exists() and start_epoch else 0),peak_reserved_bytes=max(torch.cuda.max_memory_reserved() if device=='cuda' else 0,state.get('peak_reserved_bytes',0) if path.exists() and start_epoch else 0)))
   write('TRAINING_STATUS.json',dict(model=name,fold=fold,completed_epochs=epoch+1,total_epochs=cfg['epochs'],status='RUNNING',remaining_seconds=budget.remaining()))
   print(f'{name} fold={fold} epoch={epoch+1}/{cfg["epochs"]} seconds={dt:.2f}',flush=True)
  model.eval();rows=[]
  with torch.no_grad():
   for i in te:
    prob=float(torch.sigmoid(model(*tensors(prepared[i],device),chunk=chunk)).cpu());rows.append(dict(participant_id=p.participant_id[i],label=int(p.label[i]),fold=fold,probability=prob))
  pred_path=LOCAL/f'{name}_fold{fold}_predictions.csv'
  if pred_path.exists():
   old=pd.read_csv(pred_path,dtype={'participant_id':str});assert np.allclose(old.probability,[r['probability'] for r in rows],rtol=0,atol=1e-7)
  else:table(pred_path.name,rows,local=True)
  out+=rows;table(f'{name}_oof.csv',out,local=True)
  write(f'{name}_fold{fold}_runtime.json',dict(wall_seconds=elapsed,cpu_seconds=cpu_elapsed,device=device,epochs=cfg['epochs'],peak_allocated_bytes=load(path)['peak_allocated_bytes'],peak_reserved_bytes=load(path)['peak_reserved_bytes'],checkpoint_sha256=sha(path),checkpoint_replay=True),local=True)
 return out
