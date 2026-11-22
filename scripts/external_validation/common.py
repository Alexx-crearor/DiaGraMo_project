"""Isolated Phase 6 research I/O, budgets and integrity; no baseline imports."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/external_validation'
LOCAL=ROOT/'data/external_datasets/drotar_dobes_2020'
REPO='peet292929/Dysgraphia-detection-through-machine-learning'
COMMIT='0079fdd34fc5eff0c1c58f9689127b2c73b57f8b'


def stamp():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def target(name,local=False):
    base=LOCAL if local else OUT
    p=(base/name).resolve()
    if not p.is_relative_to(base.resolve()):raise ValueError('Output escaped designated root')
    p.parent.mkdir(parents=True,exist_ok=True)
    return p


def write(name,value,local=False):
    p=target(name,local);temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,p)


def md(name,text):target(name).write_text(text.strip()+'\n',encoding='utf-8')


def table(name,rows,columns=None,local=False):
    import pandas as pd
    frame=pd.DataFrame(rows,columns=columns)
    if not local:
        context=dict(dataset='Drotar_Dobes_2020',participant_count=120,target_definition='Workbook diag DYSGR vs0; study clinical assessment',
            split_protocol='Not a performance evaluation unless stated',model='N/A',completed_folds=0,
            metric='N/A descriptive audit',confidence_interval='N/A',evaluation_scope='Exploratory compatibility audit')
        for key,value in context.items():
            if key not in frame:frame[key]=value
    frame.to_csv(target(name,local),index=False)


def load(name,local=False):return json.loads(target(name,local).read_text(encoding='utf-8'))


class Budget:
    def __init__(self,hours=8,reserve_minutes=15):
        if hours<=0 or reserve_minutes<0 or reserve_minutes*60>=hours*3600:raise ValueError('Invalid execution budget')
        self.start=time.monotonic();self.seconds=hours*3600;self.reserve=reserve_minutes*60
        clock=OUT/'SESSION_CLOCK.json'
        if clock.exists():self.initial_epoch=json.loads(clock.read_text(encoding='utf-8'))['initial_epoch']
        else:
            hardware=OUT/'HARDWARE_BUDGET.json'
            if hardware.exists():
                d=json.loads(hardware.read_text(encoding='utf-8'))
                self.initial_epoch=datetime.fromisoformat(d['utc']).timestamp()-d['budget']['actual_seconds']
            else:self.initial_epoch=time.time()
            write('SESSION_CLOCK.json',dict(initial_epoch=self.initial_epoch,policy='Single overall session across CLI phases; resume does not reset budget'))
    def available(self):return max(0.,self.seconds-self.reserve-(time.time()-self.initial_epoch))
    def admit(self,estimate_seconds):
        if estimate_seconds<0 or estimate_seconds>self.available():raise TimeoutError('Budget gate refused operation')
    def summary(self):return dict(max_runtime_seconds=self.seconds,report_reserve_seconds=self.reserve,actual_seconds=time.time()-self.initial_epoch,remaining_experiment_seconds=self.available())


def status(phase,state,started,evidence,validation,artifacts,reason=None):
    if state not in {'COMPLETE','PARTIAL','BLOCKED','SKIPPED'}:raise ValueError('Bad stage status')
    write(f'phase{phase}_status.json',dict(phase=phase,status=state,utc=stamp(),runtime_seconds=time.monotonic()-started,
        evidence=evidence,validation=validation,artifacts=artifacts,reason=reason))


def initialize(budget):
    import platform
    import importlib.metadata
    import psutil
    import torch
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    assert subprocess.run(['git','check-ignore',str(LOCAL.relative_to(ROOT)/'private.csv')],cwd=ROOT,capture_output=True).returncode==0
    manifest=OUT/'PROTECTED_ARTIFACT_MANIFEST.json'
    if not manifest.exists():
        names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        # Include every existing persisted model, including ignored checkpoints, hash-only.
        for base in [ROOT/'results/models',ROOT/'results/kinematic_xai/checkpoints',ROOT/'results/raw_temporal_v2/cache/phase4_lite',ROOT/'data/derived/raw_temporal_v2']:
            if base.exists():names.extend(p.relative_to(ROOT).as_posix() for p in base.rglob('*') if p.is_file() and p.suffix in {'.pt','.joblib'})
        for p in (ROOT/'results/raw_temporal_v2/phase4_lite/fits').glob('*/result.json'):
            r=json.loads(p.read_text(encoding='utf-8'))
            if r.get('checkpoint'):names.append(r['checkpoint'])
        files=[]
        for name in sorted(set(names)):
            if name.startswith(('results/external_validation/','scripts/external_validation/','tests/external_validation/')):continue
            p=ROOT/name
            if not p.is_file():raise FileNotFoundError(name)
            files.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
        write('PROTECTED_ARTIFACT_MANIFEST.json',dict(reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=files))
    packages={}
    for k in ['torch','numpy','pandas','scikit-learn','lightgbm','scipy','psutil']:
        try:packages[k]=importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:packages[k]=None
    vm=psutil.virtual_memory();disk=psutil.disk_usage(str(ROOT))
    gpu=dict(cuda_available=torch.cuda.is_available(),training_processes_started=0,allocation_limit_gib=3.2)
    if gpu['cuda_available']:
        free,total=torch.cuda.mem_get_info()
        gpu.update(name=torch.cuda.get_device_name(0),total_bytes=total,free_bytes=free,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    info=dict(python=platform.python_version(),packages=packages,cpu_logical=psutil.cpu_count(),cpu_physical=psutil.cpu_count(logical=False),
        affinity_logical=len(psutil.Process().cpu_affinity()),ram_total_bytes=vm.total,ram_available_bytes=vm.available,
        disk_free_bytes=disk.free,gpu=gpu,maximum_gpu_workers=1,budget=budget.summary(),utc=stamp(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        initial_git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines())
    write('HARDWARE_BUDGET.json',info)
    contract=dict(kinematic=dict(tasks=14,features_per_task=74,participant_dimensions=1036,output='pK',
        extractor='src/diagramo/features.py; src/diagramo/branches/kinematic.py',evaluation='scripts/evaluate.py',normalization='Train-fold median imputer/constant filter/scaler'),
        temporal=dict(raw_channels=7,processed_channels=21,maximum_window_samples=128,kernels=[5,13,29],tokens_per_window=4,
        token_dimension=32,heads=2,task_embedding_count=14,participant_pool='Learned Pool',output='pT',parameters=25252,
        classes=['MultiScaleLocal','RelativeStrokeBlock','MultiScaleModel'],source='src/diagramo/temporal_v2/multiscale.py',
        preprocessing='src/diagramo/temporal/preprocessing.py; segmentation.py; dataset.py'),
        frozen=dict(seed=42,folds=6,threshold='p>0.5',epochs=28,checkpoint_policy='Fixed final epoch; no Test selection',
        fusion='0.5*pK+0.5*pT',metadata='results/raw_temporal_v2/phase4_lite/fits/*/result.json',
        config='configs/raw_temporal_v2/phase4_lite.json'),architecture_modified=False)
    write('ORIGINAL_MODEL_CONTRACT.json',contract)
    md('PHASE0_ENVIRONMENT.md',f'''# Phase 0 — Repository, hardware và model contract

Branch `{info['branch']}` tạo tại cùng commit, giữ bốn tệp untracked cũ. Không reset/stash/delete, không chạy queue cũ. Budget mặc định8giờ, reserve15phút, CLI cho phép đổi; đây là trần, không phải yêu cầu dùng hết.

CPU {info['cpu_physical']} physical/{info['cpu_logical']} logical, affinity {info['affinity_logical']}; RAM total {vm.total/2**30:.2f}GiB, available {vm.available/2**30:.2f}GiB; disk free {disk.free/2**30:.2f}GiB. CUDA available={gpu['cuda_available']}; GPU {gpu.get('name','unknown')}. Framework availability chưa phải bằng chứng forward/backward; không chạy GPU nếu chưa qua data gate.

Đã kiểm tra code Kinematic, evaluate.py, preprocessing/segmentation/dataset, MultiScaleLocal/RelativeStrokeBlock/MultiScaleModel và frozen metadata. Contract chi tiết trong ORIGINAL_MODEL_CONTRACT.json. 14×74=1036; Temporal7→21channels, windows≤128, kernels5/13/29,4tokens32dims,2heads, task embedding14 và learned participant pooling. Không có task-level Transformer. Mô hình/checkpoints/OOF chỉ đọc và hash; không deserialize checkpoint để khảo sát dataset không tương thích.

Manifest bảo vệ gồm toàn tracked tree tại mốc và các model/checkpoint đã phát hiện. Đã đọc Data Audit V3, Phase5 provenance/requirements/shortlist/decision và LOCKED_RESEARCH_REPORT_VI.md. Chỉ tạo đầu ra mới dưới external_validation và dataignored.
''')
    status(0,'COMPLETE',start,dict(protected_count=len(load('PROTECTED_ARTIFACT_MANIFEST.json')['files']),model_contract=contract),
        dict(data_directory_git_ignored=True,original_architecture_untouched=True),['HARDWARE_BUDGET.json','PROTECTED_ARTIFACT_MANIFEST.json','ORIGINAL_MODEL_CONTRACT.json','PHASE0_ENVIRONMENT.md'])
