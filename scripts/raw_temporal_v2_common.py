"""Shared, participant-isolated infrastructure for the recovery investigation.

The v1 implementation and results remain immutable. Deterministic v1 caches
are reused by reference; each new phase locks only its own relevant code.
"""
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from raw_temporal_common import cohort, recording_map, raw_svc_path, sha

OUT = ROOT / 'results/raw_temporal_v2'
OLD = ROOT / 'results/raw_temporal'
CACHE = OLD / 'cache'
LOCAL = ROOT / 'data/derived/raw_temporal_v2'
RESOURCE = json.loads((ROOT / 'configs/raw_temporal_v2/resources.json').read_text())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FIELDS = ['experiment_id', 'phase', 'hypothesis', 'model', 'input_representation',
    'ablation', 'split', 'seed', 'configuration_hash', 'source_code_hash',
    'training_status', 'best_checkpoint', 'validation_metrics', 'held_out_metrics',
    'parameter_count', 'gpu_peak_memory', 'training_cost', 'artifact_paths',
    'completion_status', 'reason', 'scientific_interpretation']


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(path.name + '.tmp')
    for attempt in range(20):
        try:
            temporary.write_text(data, encoding='utf-8', newline='\n')
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(.1)
    raise PermissionError('Cannot atomically persist ' + str(path))


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(RESOURCE['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync():
    if DEVICE == 'cuda': torch.cuda.synchronize()


def metrics(labels, probabilities):
    y = np.asarray(labels); p = np.asarray(probabilities); prediction = p > .5
    return dict(accuracy=float(accuracy_score(y, prediction)),
        f1_macro=float(f1_score(y, prediction, average='macro', zero_division=0)),
        auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        pr_auc=float(average_precision_score(y, p)) if y.sum() else None,
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        sensitivity=float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        specificity=float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        brier=float(brier_score_loss(y, p)))


def development():
    ids, y, schema, assignments, outer = cohort()
    cfg = json.loads((ROOT / 'configs/raw_temporal_v2/phase0.json').read_text())
    train, test = outer[cfg['development_outer_fold']]
    inner = list(StratifiedKFold(3, shuffle=True, random_state=42).split(train, y[train]))
    a, b = inner[cfg['development_inner_fold']]
    fit, validation = train[a], train[b]
    assert not set(fit) & set(validation) and not set(train) & set(test)
    return ids, y, schema, fit, validation, test


def balanced_tiny(size):
    ids, y, _, fit, _, _ = development()
    rng = np.random.default_rng(42)
    chosen = np.r_[rng.permutation(fit[y[fit] == 0])[:size // 2],
                   rng.permutation(fit[y[fit] == 1])[:size - size // 2]]
    return np.sort(chosen)


def signature(config, scripts, modules=()):
    paths = ['scripts/raw_temporal_v2_common.py', 'configs/raw_temporal_v2/resources.json',
             config] + list(scripts) + list(modules)
    # The exact old preprocessing/encoder code reused by a phase is included.
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'src/diagramo/temporal').glob('*.py')]
    values = {name: sha(ROOT / name) for name in sorted(set(paths))}
    data = {name: sha(ROOT / name) for name in [
        'results/cv/fold_assignments.csv', 'data/derived/schema.json',
        'results/raw_temporal/cache/manifest.json',
        'results/raw_temporal/audit/source_hashes.json']}
    code = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    whole = hashlib.sha256(json.dumps(dict(code=values, data=data), sort_keys=True).encode()).hexdigest()
    return dict(signature=whole, source_code_hash=code, source_files=values,
                configuration_hash=sha(ROOT / config), data_hashes=data)


def resource_check():
    free = shutil.disk_usage(ROOT).free
    assert free >= RESOURCE['minimum_free_storage_gib'] * 1024**3, 'Storage budget exhausted'
    if DEVICE == 'cuda':
        assert torch.cuda.max_memory_allocated() <= RESOURCE['gpu_tensor_limit_mib'] * 1024**2, 'Tensor memory budget exceeded'
    return free


@contextlib.contextmanager
def gpu_worker():
    """Exclusive scheduler lock. Stale locks are recoverable after interruption."""
    import psutil
    LOCAL.mkdir(parents=True, exist_ok=True)
    path = LOCAL / 'gpu_worker.lock'
    if path.exists():
        prior = json.loads(path.read_text())
        if psutil.pid_exists(prior['pid']):
            process = psutil.Process(prior['pid'])
            if abs(process.create_time() - prior['created']) < .01:
                raise RuntimeError('Another recovery GPU worker is active')
        path.unlink()
    with path.open('x') as handle:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), handle)
    try:
        resource_check()
        yield
    finally:
        path.unlink(missing_ok=True)


def ledger(experiment_id, **updates):
    path = OUT / 'EXPERIMENT_LEDGER.csv'; path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(path.open(encoding='utf-8', newline=''))) if path.exists() else []
    matches = [r for r in rows if r['experiment_id'] == experiment_id]
    assert len(matches) <= 1
    row = matches[0] if matches else dict.fromkeys(FIELDS, '')
    if not matches:
        row['experiment_id'] = experiment_id; rows.append(row)
    for key, value in updates.items():
        assert key in FIELDS, key
        row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) if isinstance(value, (dict,list)) else str(value)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def begin(experiment_id, phase, model, sig, **metadata):
    path = OUT / 'EXPERIMENT_LEDGER.csv'
    rows = list(csv.DictReader(path.open(encoding='utf-8', newline=''))) if path.exists() else []
    started = [r for r in rows if r['training_status'] in ('running','completed','failed')]
    existing = [r for r in rows if r['experiment_id'] == experiment_id]
    if existing:
        prior = existing[0]
        if prior['source_code_hash']:
            assert prior['source_code_hash'] == sig['source_code_hash'], 'Incompatible code resume: ' + experiment_id
            assert prior['configuration_hash'] == sig['configuration_hash'], 'Incompatible config resume'
    if (not existing or not existing[0]['source_code_hash']) and len(started) >= RESOURCE['maximum_training_runs']:
        ledger(experiment_id, phase=phase, model=model, training_status='skipped',
               completion_status='not_executed', reason='Configured training run budget exhausted')
        raise RuntimeError('Configured training run budget exhausted')
    ledger(experiment_id, phase=phase, model=model, source_code_hash=sig['source_code_hash'],
           configuration_hash=sig['configuration_hash'], training_status='running',
           completion_status='in_progress', **metadata)


def setup():
    OUT.mkdir(parents=True, exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True)
    for directory in ['phase0_diagnostics','phase1_bottleneck','phase2_temporal',
                      'phase3_ssl','phase4_complementarity','tables','figures','checkpoints_manifest']:
        (OUT / directory).mkdir(exist_ok=True)
    start = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    protected = OUT / 'protected_git_tree.json'
    if not protected.exists():
        lines = subprocess.check_output(['git','ls-tree','-r','HEAD'], cwd=ROOT, text=True).splitlines()
        write(protected, {line.split('\t',1)[1]: line.split()[2] for line in lines})
        write(OUT / 'provenance.json', dict(start_head=start,
            branch='study/raw-temporal-research-v2', original_cache=str(CACHE.relative_to(ROOT)).replace('\\','/'),
            python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            historical_outer_inspection=True, inference_scope='Exploratory internal participant CV',
            protected_original_raw_results=True, remote_writes_authorized=False))
    ids,y,schema,fit,val,test = development()
    roles = np.full(len(ids), 'outer_test_excluded', dtype=object)
    roles[fit]='development_fit'; roles[val]='development_validation'
    pd.DataFrame(dict(participant_id=ids, role=roles)).to_csv(OUT/'development_assignments.csv',index=False)
    # Labels for the designated Test are deliberately absent from development artifacts.
    write(OUT/'validation.json', dict(complete=False, phase0='in_progress', phase1='planned',
        phase2='planned', phase3='planned', phase4='planned',
        external_evaluation_status='not_yet_audited', unfinished=['Phase 0–4 execution and validation']))
    print('RECOVERY WORKSPACE INITIALIZED',len(fit),len(val),len(test),flush=True)


if __name__ == '__main__': setup()
