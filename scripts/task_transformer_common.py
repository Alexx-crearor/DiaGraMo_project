"""Shared immutable inputs/provenance for the Kinematic Transformer study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from kinematic_xai_common import load_data, sha, write, factory, metrics
from diagramo.kinematic_xai import feature_mapping

OUT = ROOT / 'results/task_transformer'
CONFIG_PATH = ROOT / 'configs/task_transformer.json'
CONFIG = json.loads(CONFIG_PATH.read_text())
FAMILIES = CONFIG['families']
METRICS = ('f1_macro', 'accuracy', 'auc', 'balanced_accuracy')


def data():
    return load_data()


def score(y, probability, prediction=None):
    return {k: v for k, v in metrics(y, (np.asarray(probability) > .5).astype(int) if prediction is None else prediction, probability).items() if k in METRICS}


def device():
    torch.set_num_threads(4)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize():
    if (OUT / 'config.json').exists():
        assert json.loads((OUT / 'config.json').read_text()) == CONFIG
        return
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [p for p in (ROOT / 'results').rglob('*') if p.is_file() and not p.is_relative_to(OUT)]
    for folder in ('src/diagramo/branches', 'notebooks'):
        protected += [p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected += [ROOT / p for p in ('data/derived/kinematic.npz', 'data/derived/schema.json',
                   'src/diagramo/features.py', 'src/diagramo/kinematic_xai.py', 'scripts/evaluate.py',
                   'scripts/prepare.py', 'scripts/kinematic_xai_common.py', 'EXPERIMENT_PROTOCOL.md', 'requirements-lock.txt')]
    write(OUT / 'protected_manifest.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(protected))})
    write(OUT / 'config.json', CONFIG)
    write(OUT / 'provenance.json', dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
          branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip(),
          status_before=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT).decode(),
          torch=torch.__version__, numpy=np.__version__, device=device(),
          gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          note='Task-1 outer-test XAI has motivated hypotheses; reused cohort evaluation remains exploratory.',
          reused=['original raw-record Kinematic extractor', 'saved 6 folds and LightGBM',
                  'Task-1 train_only_task_importance API', 'Task-1 validated outer-Train rankings when IDs/input hashes match']))
    write(OUT / 'validation.json', dict(complete=False, stage='initialized'))


def cache():
    return np.load(OUT / 'cache/features.npz')


def features_for(family, bank, condition='clean'):
    if family == 'motor_normalized':
        return bank['motor_clean' if condition == 'clean' else 'motor_stress_' + condition]
    return bank['raw_clean' if condition == 'clean' else 'stress_' + condition]


def training_signature():
    paths = ['configs/task_transformer.json', 'src/diagramo/task_transformer.py',
             'scripts/task_transformer_common.py', 'scripts/train_task_transformer.py']
    signature = {p: sha(ROOT / p) for p in paths}
    p = OUT / 'training_code.json'
    if p.exists():
        assert json.loads(p.read_text()) == signature, 'Training code changed; choose a separate result directory'
    else:
        write(p, signature)
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
