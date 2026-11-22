import json,time,subprocess
import numpy as np
import pandas as pd
from .common import *
from .evaluate import complete_oof
from scripts.external_validation.experiments import metrics

def bootstrap(y,probabilities,reps=2000):
 rng=np.random.default_rng(42);keys=['f1_macro','auc','balanced_accuracy','accuracy','sensitivity','specificity','brier','log_loss'];values={name:{k:[] for k in keys} for name in probabilities}
 for _ in range(reps):
  at=rng.integers(0,len(y),len(y))
  for name,p in probabilities.items():
   score=metrics(y[at],p[at])
   for k in keys:
    if score[k] is not None:values[name][k].append(score[k])
 return values

def generate(p,cfg,budget):
 frames={};registry=[];folds=[];runtime=[];checkpoints=[];incomplete={}
 for name in cfg['models']:
  path=LOCAL/f'{name}_oof.csv'
  f=pd.read_csv(path,dtype={'participant_id':str}) if path.exists() else pd.DataFrame()
  n=len(f);completed=int(f.fold.nunique()) if n else 0
  if n==120:
   f=complete_oof(f);assert f.participant_id.tolist()==p.participant_id.tolist() and (f.label==p.label).all() and (f.fold==p.candidate_fold).all()
   assert np.isfinite(f.probability).all();frames[name]=f
  else:incomplete[name]=dict(participants=n,folds=completed)
  registry.append(dict(model=name,status='COMPLETE' if n==120 else 'PARTIAL' if n else 'NOT_RUN',completed_folds=completed,participants=n,expected_participants=120,seed=42,interpretation='Exploratory internal participant CV; not source-independent validation'))
  if n:
   for fold,g in f.groupby('fold'):folds.append(dict(model=name,fold=int(fold),participants=len(g),**metrics(g.label,g.probability)))
  for fold in range(5):
   rp=LOCAL/f'{name}_fold{fold}_runtime.json'
   if rp.exists():runtime.append(dict(model=name,fold=fold,**json.loads(rp.read_text())))
   paths=list((LOCAL/'checkpoints').glob(f'{name}_fold{fold}.*'))
   for checkpoint in paths:
    if checkpoint.suffix in {'.pt','.joblib'}:checkpoints.append(dict(alias=f'{name}_fold{fold}',sha256=sha(checkpoint),storage='local-only',type=checkpoint.suffix,completed_fold=bool(n and fold in set(f.fold))))
 y=p.label.to_numpy();prob={k:f.probability.to_numpy() for k,f in frames.items()};scores={k:metrics(y,v) for k,v in prob.items()}
 boot=bootstrap(y,prob,cfg['bootstrap_replicates'])
 rows=[];uncertainty=[]
 for name,score in scores.items():
  row=dict(model=name,participants=120,completed_folds=5,**score)
  own=[r for r in folds if r['model']==name]
  for k in boot[name]:
   lo,hi=np.quantile(boot[name][k],[.025,.975]);row[k+'_ci95_low']=float(lo);row[k+'_ci95_high']=float(hi)
   row[k+'_fold_mean']=float(np.mean([r[k] for r in own]));row[k+'_fold_sd']=float(np.std([r[k] for r in own],ddof=1))
   uncertainty.append(dict(model=name,metric=k,low=float(lo),high=float(hi),bootstrap_replicates=cfg['bootstrap_replicates'],resampling_unit='Participant fixed OOF; matched indices; excludes refitting, seed, site and selection uncertainty'))
  rows.append(row)
 pairs=[];overlap=[]
 for left,right in [('B3','B4'),('B2','B1'),('B3','B2'),('B1','B0'),('B2','B0')]:
  if left not in frames or right not in frames:continue
  a=prob[left]>.5;b=prob[right]>.5;ac=a==y;bc=b==y
  row=dict(left=left,right=right,matched_participants=120,corrected=int((ac&~bc).sum()),harmed=int((~ac&bc).sum()),both_correct=int((ac&bc).sum()),both_wrong=int((~ac&~bc).sum()),prediction_disagreements=int((a!=b).sum()),mean_probability_difference=float(np.mean(prob[left]-prob[right])),mean_absolute_probability_difference=float(np.mean(np.abs(prob[left]-prob[right]))))
  for k in ['f1_macro','auc','balanced_accuracy','accuracy']:
   delta=np.array(boot[left][k])-np.array(boot[right][k]);lo,hi=np.quantile(delta,[.025,.975]);row[k+'_difference']=scores[left][k]-scores[right][k];row[k+'_ci95_low']=float(lo);row[k+'_ci95_high']=float(hi)
  pairs.append(row);overlap.append({k:v for k,v in row.items() if k in ['left','right','matched_participants','corrected','harmed','both_correct','both_wrong','prediction_disagreements']})
 table('MODEL_METRICS.csv',rows);table('FOLD_METRICS.csv',folds);table('EXPERIMENT_REGISTRY.csv',registry);table('RUNTIME_GPU_MEMORY.csv',runtime);table('PAIRED_MODEL_COMPARISON.csv',pairs);table('ERROR_OVERLAP.csv',overlap);table('UNCERTAINTY_ANALYSIS.csv',uncertainty)
 write('CHECKPOINT_MANIFEST.json',dict(checkpoints=checkpoints,prior_control='Reuse Phase6 hashed five JSON checkpoints'))
 # Descriptive recording-length error association with fixed quartiles, no selection.
 from .participant_dataset import load_data
 _,data=load_data();length=np.array([sum(len(x) for x in r) for r in data]);edges=np.quantile(length,[.25,.5,.75]);bins=np.searchsorted(edges,length,side='right');error_rows=[]
 from scipy.stats import spearmanr
 for name,pr in prob.items():
  for q in range(4):
   at=bins==q;error_rows.append(dict(model=name,length_quartile=q+1,participants=int(at.sum()),errors=int(((pr>.5)!=y)[at].sum()),class1_count=int(y[at].sum()),mean_probability=float(pr[at].mean()),sample_count_min=int(length[at].min()),sample_count_max=int(length[at].max()),probability_length_spearman=float(spearmanr(length,pr).statistic) if np.std(pr)>0 else None))
 table('ERROR_BY_LENGTH.csv',error_rows)
 # Protected files and source hash checks: no model unpickling outside isolated new checkpoints.
 manifest=read('PROTECTED_INPUT_MANIFEST.json');changed=[r['path'] for r in manifest['files'] if sha(ROOT/r['path'])!=r['sha256']]
 source=read('protected_source_files.json',True);source_changed=sum(sha(ROOT/r['path'])!=r['sha256'] for r in source)
 tests=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest','tests/native_channel_research','tests/external_validation','-q'],capture_output=True,text=True)
 md('TEST_STDOUT.txt',tests.stdout+tests.stderr);write('TEST_RESULTS.json',dict(returncode=tests.returncode,utc=time.time()))
 public_text='\n'.join(x.read_text(encoding='utf8',errors='replace') for x in OUT.glob('*') if x.suffix in {'.md','.json','.csv','.txt'} and x.name not in {'PROTECTED_INPUT_MANIFEST.json','ARTIFACT_MANIFEST.json','VALIDATION_CHECKS.json'})
 # Strict standalone-token scan; generic numerical values aren't treated as IDs.
 import re
 leaks=[pid for pid in p.participant_id if re.search(r'(?<![A-Za-z0-9.])'+re.escape(pid)+r'(?![A-Za-z0-9.])',public_text)]
 tracked=subprocess.check_output(['git','ls-files','data/native_channel_research_local','data/external_datasets'],text=True).splitlines()
 validation=dict(tests_returncode=tests.returncode,protected_files_checked=len(manifest['files']),protected_files_changed=changed,private_source_files_checked=len(source),private_source_files_changed=source_changed,config_unchanged=sha(OUT/'LOCKED_CONFIG.json')==read('PREEXPERIMENT_DECISIONS.json')['config_sha256'],completed_oof_models=list(frames),incomplete_models=incomplete,all_aggregate_metrics_require_full_oof=True,participant_folds_reused=True,no_outer_test_selection=True,matched_B3_B4_architecture=True,private_files_tracked=tracked,public_identifier_leaks=leaks,cpu_exact_optimizer_rng_resume_test=True,gpu_cpu_tolerance=1e-4,original_model_transfer=False,source_independent_validation=False)
 validation['cuda_exact_optimizer_rng_resume_test']=tests.returncode==0 and read('HARDWARE_STATUS.json')['cuda_available']
 validation['valid']=not changed and not source_changed and tests.returncode==0 and not leaks and not tracked and validation['config_unchanged']
 write('VALIDATION_CHECKS.json',validation)
 primary=next((r for r in pairs if r['left']=='B3' and r['right']=='B4'),None)
 if primary is None:decision='E' if not any(k in frames for k in ['B1','B2','B3','B4']) else 'D'
 elif scores['B2']['f1_macro']>=scores['B3']['f1_macro']:decision='C'
 elif primary['f1_macro_ci95_low']>0 and sum(r['f1_macro']>next(q['f1_macro'] for q in folds if q['model']=='B4' and q['fold']==r['fold']) for r in folds if r['model']=='B3')>=4:decision='A'
 elif scores['B3']['f1_macro']>scores['B2']['f1_macro'] and primary['f1_macro_ci95_low']<=0:decision='B'
 else:decision='D'
 complete=len(frames)==5 and validation['valid']
 status(6,'COMPLETE' if len(frames)==5 else 'PARTIAL',completed_models=list(frames),incomplete=incomplete)
 status(7,'COMPLETE' if primary else 'PARTIAL');status(8,'BLOCKED',reason='Exact physical exporter schema/units still UNVERIFIED; opaque benchmark remains admitted')
 status(9,'COMPLETE',scientific_decision=decision);status(10,'COMPLETE' if validation['valid'] else 'PARTIAL')
 write('TRAINING_STATUS.json',dict(status='COMPLETE' if complete else 'PARTIAL',complete_models=list(frames),incomplete_models=incomplete,full_matched_suite_completed=len(frames)==5))
 sessions=read('EXECUTION_SESSIONS.json') if (OUT/'EXECUTION_SESSIONS.json').exists() else []
 sessions.append(dict(finished_epoch=time.time(),session_wall_seconds=budget.limit-900-budget.remaining(),completed_models=list(frames),all_tests_passed=tests.returncode==0))
 write('EXECUTION_SESSIONS.json',sessions)
 write('OVERNIGHT_RUN_STATUS.json',dict(status='COMPLETE_OPAQUE_BENCHMARK' if complete else 'PARTIAL',overall_research_complete=False,opaque_suite_complete=complete,semantic_study='BLOCKED',session_wall_seconds=budget.limit-900-budget.remaining(),remaining_experiment_seconds=budget.remaining(),scientific_decision=decision,gpu_workers_max=1,automatic_push=False))
 summary='| Model | Macro-F1 | AUC | Balanced Accuracy | Accuracy |\n|---|---:|---:|---:|---:|\n'+''.join(f"| {k} | {v['f1_macro']:.6f} | {v['auc']:.6f} | {v['balanced_accuracy']:.6f} | {v['accuracy']:.6f} |\n" for k,v in scores.items())
 cost=sum(r.get('wall_seconds',0) for r in runtime if r['model'] in ['B3','B4']);peak=max((r.get('peak_allocated_bytes',0) for r in runtime),default=0)
 from .vietnamese import write_reports
 write_reports(scores,primary,decision,complete,cost,peak,validation,summary)
 assert validation['valid'],validation
 files=[]
 for directory in [OUT,ROOT/'scripts/native_channel_research',ROOT/'tests/native_channel_research']:
  for f in sorted(directory.rglob('*')):
   if f.is_file() and '__pycache__' not in f.parts and f.name!='ARTIFACT_MANIFEST.json':files.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=sha(f),bytes=f.stat().st_size))
 write('ARTIFACT_MANIFEST.json',dict(files=files))
 print(json.dumps(dict(status='COMPLETE' if complete else 'PARTIAL',scores=scores,primary_comparison=primary,decision=decision),indent=2),flush=True)
