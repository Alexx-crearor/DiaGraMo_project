import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,sys,time,json,subprocess
import numpy as np
from pathlib import Path
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='scripts.native_channel_research'
from .common import *
from .participant_dataset import load_data
from .setup import initialize,schema,lock
from .train import benchmark,cpu_baseline,cnn

def main(argv=None):
 """Run the existing protocol; optional argv also supports imported CLI callers."""
 ap=argparse.ArgumentParser();ap.add_argument('--max-runtime-hours',type=float,default=8);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args(argv);budget=Budget(args.max_runtime_hours)
 p,data=load_data();initialize(p,data);schema(data);cfg=lock()
 if args.prepare_only:return
 tests=subprocess.run([sys.executable,'-m','pytest','tests/native_channel_research','-q'],capture_output=True,text=True)
 md('TEST_STDOUT.txt',tests.stdout+tests.stderr);write('TEST_RESULTS.json',dict(returncode=tests.returncode,utc=time.time()))
 if tests.returncode:raise RuntimeError('Unit tests failed before scoring')
 status(5,'COMPLETE',test_returncode=0)
 admission=benchmark(p,data,cfg);cnn_admitted=admission['full_B3_B4_estimated_seconds_with_2x_margin']<=budget.remaining()
 write('SUITE_ADMISSION.json',dict(cnn_admitted=cnn_admitted,remaining_seconds=budget.remaining(),estimated_seconds=admission['full_B3_B4_estimated_seconds_with_2x_margin']))
 status(3,'COMPLETE');write('normalization_leakage_checks.json',dict(train_only=True,equal_participant_weight=True,tests_returncode=0));write('window_coverage_checks.json',dict(full_coverage=True,no_cross_recording=True,padding_mask=True,terminal_samples_preserved=True,tests_returncode=0));write('preprocessing_manifest.json',dict(reader='Unmodified Phase6 load_svc_numeric',normalization=cfg['normalization'],window_size=256,permutation=cfg['permutation']))
 import pandas as pd
 prior=pd.read_csv(SOURCE/'prior_oof.csv',dtype={'participant_id':str});prior=prior.sort_values('participant_id').reset_index(drop=True)
 assert prior.participant_id.tolist()==p.participant_id.tolist() and (prior.fold==p.candidate_fold).all() and (prior.label==p.label).all()
 for f in range(5):
  state=json.loads((SOURCE/f'checkpoints/prior_fold{f}.json').read_text());sig=json.loads((SOURCE/f'checkpoints/prior_fold{f}.sha256.json').read_text());assert sha(SOURCE/f'checkpoints/prior_fold{f}.json')==sig['sha256']
  assert np.all(prior[prior.fold==f].probability==state['probability'])
 table('B0_oof.csv',prior.to_dict('records'),local=True)
 try:
  cpu_baseline('B1',p,data,cfg,budget);cpu_baseline('B2',p,data,cfg,budget)
  for name in (['B3','B4'] if cnn_admitted else []):
   for chunk,force_cpu in [(64,False),(32,False),(16,False),(8,False),(8,True)]:
    try:cnn(name,p,data,cfg,budget,chunk,force_cpu);break
    except RuntimeError as e:
     if 'out of memory' not in str(e).lower() and 'allocation budget' not in str(e).lower():raise
     import torch,gc
     write(f'{name}_fallback_{chunk}_{force_cpu}.json',dict(error=str(e),chunk=chunk,force_cpu=force_cpu,checkpoint_policy='Restart interrupted epoch from last hashed checkpoint'))
     gc.collect();torch.cuda.empty_cache()
     if force_cpu:raise
 except TimeoutError as e:write('BUDGET_STOP.json',dict(reason=str(e),remaining_seconds=budget.remaining()))
 except Exception as e:
  write('FAILURE_STATUS.json',dict(error_type=type(e).__name__,message=str(e),status='PARTIAL',checkpoint_policy='Preserved; investigate before resume'))
  from .evaluate import finalize
  finalize(p,cfg,budget)
  raise
 from .evaluate import finalize
 finalize(p,cfg,budget)
if __name__=='__main__':
 with worker_lock():main()
