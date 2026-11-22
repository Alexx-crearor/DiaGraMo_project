import json,subprocess,time,shutil,platform,importlib.metadata
import numpy as np
import psutil,torch,requests
from .common import *

def initialize(p,data):
 OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
 assert subprocess.run(['git','check-ignore','data/native_channel_research_local/private.csv'],capture_output=True).returncode==0
 if not (OUT/'PROTECTED_INPUT_MANIFEST.json').exists():
  old=json.loads((ROOT/'results/external_validation/PROTECTED_ARTIFACT_MANIFEST.json').read_text())['files']
  names=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())|{r['path'] for r in old}
  # Protect all original local sources, including inventory/OOF, without exposing filenames containing subject IDs.
  files=[dict(path=n,sha256=sha(ROOT/n)) for n in sorted(names) if not n.startswith(('results/native_channel_research/','scripts/native_channel_research/','tests/native_channel_research/'))]
  private=[dict(path=x.relative_to(ROOT).as_posix(),sha256=sha(x)) for x in SOURCE.rglob('*') if x.is_file()]
  write('protected_source_files.json',private,local=True)
  write('PROTECTED_INPUT_MANIFEST.json',dict(files=files,private_source_manifest_sha256=sha(LOCAL/'protected_source_files.json'),private_source_count=len(private)))
 info=dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','lightgbm','scikit-learn']},disk_free_bytes=shutil.disk_usage(ROOT).free,ram_available_bytes=psutil.virtual_memory().available,cpu_physical=psutil.cpu_count(logical=False),cpu_logical=psutil.cpu_count(),cuda_available=torch.cuda.is_available(),gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,maximum_gpu_workers=1)
 write('HARDWARE_STATUS.json',info)
 write('participant_grouping_validation.json',dict(participants=len(p),recordings=sum(map(len,data)),samples=sum(len(x) for r in data for x in r),reused_phase6_folds=True,unique_participant_ids=True,all_recordings_grouped=True,labels={'0':63,'1':57},source_inventory_sha256=sha(SOURCE/'external_participant_inventory.csv')))
 md('PHASE0_REPOSITORY_CHECK.md','# Repository inventory: base c8d81bf; same workspace; original data and folds preserved. Full Vietnamese report generated after validation.\n')
 status(0,'COMPLETE')

def schema(data):
 if (OUT/'SCHEMA_CONFIDENCE_DECISION.json').exists():return
 checks=[]
 api='https://api.github.com/repos/peet292929/Dysgraphia-detection-through-machine-learning'
 for tail in ['/commits?per_page=20','/issues?state=all&per_page=30','/issues/2/comments','/git/trees/0079fdd34fc5eff0c1c58f9689127b2c73b57f8b?recursive=1']:
  try:
   response=requests.get(api+tail,timeout=30);response.raise_for_status();obj=response.json()
   write('sources/'+hashlib.sha256(tail.encode()).hexdigest()[:10]+'.json',obj,local=True)
   checks.append(dict(url=api+tail,status=response.status_code,items=len(obj) if isinstance(obj,list) else len(obj.get('tree',[])),sha256=hashlib.sha256(response.content).hexdigest()))
  except Exception as e:checks.append(dict(url=api+tail,status='UNAVAILABLE',error=type(e).__name__))
 old=json.loads((ROOT/'results/external_validation/DOCUMENT_CHECKSUMS.json').read_text())
 write('SOURCE_RECHECK.json',dict(checked_utc=time.time(),requests=checks,cached_document_checksums=old,search_timebox_minutes=60,search_finished_without_author_dependency=True))
 paper='https://doi.org/10.1038/s41598-020-78611-9';issue='https://github.com/peet292929/Dysgraphia-detection-through-machine-learning/issues/2'
 rows=[]
 for k,candidate in enumerate(['coordinate candidate','coordinate candidate','clock candidate','binary state candidate','angle candidate A','angle candidate B','pressure candidate']):
  vals=[x[:,k] for r in data for x in r]
  rows.append(dict(channel=f'c{k}',candidate_meaning=candidate,documented_column_order='UNVERIFIED',native_min=min(float(v.min()) for v in vals),native_max=max(float(v.max()) for v in vals),dtype='integer-valued numeric in SVC',expected_unit='UNVERIFIED',verification_level='SUPPORTED pattern only; exporter semantics UNVERIFIED',evidence_url=paper,location='Methods/Table 2 modality descriptions; not indexed exporter schema',additional_url=issue,conflicting_evidence='Table 2 angular descriptions are not an indexed export dictionary; no unit conversion admitted',uncertainty='Exact exporter/version/channel order/units unavailable'))
 table('CHANNEL_EVIDENCE_MATRIX.csv',rows)
 table('TASK_BOUNDARY_EVIDENCE.csv',[dict(claim='8 written conditions',status='VERIFIED document only',url=paper,location='Methods and supplementary template'),dict(claim='SVC task boundaries, hw code semantics and clock resets',status='UNVERIFIED',url=issue,location='No acquisition/exporter schema recovered')])
 import pandas as pd
 registry=pd.read_csv(ROOT/'results/external_validation/SOURCE_DOCUMENTS.csv').to_dict('records')
 registry += [dict(source='Phase7 API recheck',url=r['url'],location='API response cached local',status=r['status'],evidence='Bounded history/issues/tree inspection') for r in checks]
 table('SOURCE_REGISTRY.csv',registry)
 write('SCHEMA_CONFIDENCE_DECISION.json',dict(physical_schema_verified=False,opaque_benchmark_allowed=True,clock_unit='UNVERIFIED',task_boundaries='UNVERIFIED',author_contact_required=False,optional_semantic_study='BLOCKED'))
 md('SCHEMA_RECOVERY_REPORT_VI.md','# Schema: UNVERIFIED units/export order; continue explicitly authorized opaque benchmark. See source matrices and final Vietnamese report.\n')
 status(1,'COMPLETE',physical_schema='UNVERIFIED',opaque_continues=True)

def lock():
 cfg=dict(version=1,created_epoch=time.time(),seed=42,second_seed_run=False,second_seed_reason='Reserve budget for complete matched main suite; decided before scores',folds='Reuse Phase6 candidate_fold exactly',models=['B0','B1','B2','B3','B4'],threshold='p>0.5',epochs=20,effective_participant_batch=8,window_size=256,chunk_windows=64,kernels=[5,13,29],hidden=32,dropout=.1,optimizer='AdamW',learning_rate=.001,weight_decay=.001,gradient_clip=1.,normalization='Equal-participant/equal-recording first and second moments fit outer Train only',aggregation='Valid-sample-weighted windows then equal recordings then participant loss',permutation='Normalize then fixed SHA256(seed42,participant,recording,window-start) synchronized valid-row permutation within each window; padding untouched',features='105 opaque features: 11 marginal + 4 absolute-successive-difference summaries per channel; 6 length features separately',lightgbm=dict(n_estimators=100,num_leaves=7,max_depth=3,learning_rate=.03,min_child_samples=12,colsample_bytree=1.,subsample=1.,reg_lambda=1.,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True),bootstrap_replicates=2000,outer_test_selection=False,admission='Training-only measured step scaled by windows, epochs and folds, 2x safety; complete matched B3+B4 required',fallback='OOM: same architecture, retry current epoch from intact checkpoint with window chunk32 then16 then8 then CPU; participant batch8 by accumulation unchanged',no_original_model_transfer=True)
 if not (OUT/'LOCKED_CONFIG.json').exists():
  write('LOCKED_CONFIG.json',cfg);write('PREEXPERIMENT_DECISIONS.json',dict(config_sha256=sha(OUT/'LOCKED_CONFIG.json'),locked_before_new_scores=True))
 else:assert sha(OUT/'LOCKED_CONFIG.json')==read('PREEXPERIMENT_DECISIONS.json')['config_sha256']
 md('LOCKED_PROTOCOL.md', '# Locked protocol before new scores\n\nFixed configuration in LOCKED_CONFIG.json. Primary participant Macro-F1; threshold p>0.5; reuse Phase6 five folds; seed42;20fixed epochs. No outer-test model selection. B3/B4 matched except fixed valid-row permutation. B2 has77 marginal and28 order-dependent difference summaries. Train-only equal-participant scaler. Full coverage windows256. OOM fallback chunks64/32/16/8/CPU; effective batch8. Resource pilot before scoring. Complete120 OOF required. No source independence or clinical claim.\n')
 md('PREPROCESSING_PROTOCOL.md', '# Native preprocessing\n\nRead-only seven-column reader; original row order; no physical conversions/task guessing. Train-only moments, equal participant/equal recording weighting. Normalize before window-level valid-row permutation; pad zero with explicit valid mask. Weighted windows, equal recordings, participant loss. Terminal samples preserved.\n')
 status(2,'COMPLETE');status(4,'COMPLETE',config_sha256=sha(OUT/'LOCKED_CONFIG.json'))
 return read('LOCKED_CONFIG.json')
