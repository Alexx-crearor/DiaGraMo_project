"""Fixed inputs and provenance for the isolated Kinematic task-level XAI study."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
OUT = ROOT / 'results/kinematic_xai'
SEED = 42
REPEATS = 30
METRICS = ('f1_macro', 'accuracy', 'auc', 'balanced_accuracy')
import numpy as np
import pandas as pd
from evaluate import classifiers, build_pipeline, metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def factory():
    return build_pipeline('LightGBM', classifiers(SEED)['LightGBM'])


def load_data():
    d = np.load(ROOT / 'data/derived/kinematic.npz')
    X, y, ids = d['X'], d['y'], d['ids']
    schema = json.loads((ROOT / 'data/derived/schema.json').read_text())
    assignments = pd.read_csv(ROOT / 'results/cv/fold_assignments.csv')
    oof = pd.read_csv(ROOT / 'results/cv/kinematic__LightGBM.oof.csv', float_precision='round_trip')
    if X.shape != (258, 1036) or len(set(ids)) != 258 or not np.isfinite(X).all():
        raise ValueError('Unexpected Kinematic cohort or features')
    for frame in (assignments, oof):
        assert np.array_equal(ids, frame.participant_id) and np.array_equal(y, frame.label)
    assert np.array_equal(assignments.fold, oof.fold)
    splits = [(np.flatnonzero(assignments.fold.to_numpy() != f), np.flatnonzero(assignments.fold.to_numpy() == f)) for f in range(6)]
    from sklearn.model_selection import StratifiedKFold
    for (tr, te), (original_tr, original_te) in zip(splits, StratifiedKFold(6, shuffle=True, random_state=42).split(X, y)):
        assert np.array_equal(tr, original_tr) and np.array_equal(te, original_te)
    settings = json.loads((ROOT / 'results/cv/settings.json').read_text())
    assert settings['seed'] == 42 and settings['label'] == 'group'
    assert settings['classifiers']['LightGBM'] == classifiers(42)['LightGBM'].get_params()
    return X, y, ids, schema, assignments, oof, splits


def initialize():
    config = dict(seed=SEED, outer_folds=6, participants=258, features=1036,
                  permutation_repeats=REPEATS, shap_output='raw_log_odds',
                  shap_perturbation='tree_path_dependent',
                  test_time_removal='all 74 raw columns replaced by outer-Train medians',
                  retraining_ablation='14 prespecified Leave-One-Task-Out; fixed original pipeline',
                  train_only='3-fold cross-fitted importance within supplied Train rows; no subset selection')
    if (OUT / 'config.json').exists():
        assert json.loads((OUT / 'config.json').read_text()) == config
        for rel, digest in json.loads((OUT / 'baseline_manifest.json').read_text()).items():
            assert sha(ROOT / rel) == digest, rel
        return config
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [p for p in (ROOT / 'results').rglob('*') if p.is_file() and not p.is_relative_to(OUT)]
    protected += [p for folder in ('notebooks', 'src/diagramo/branches') for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected += [ROOT / rel for rel in ('data/derived/kinematic.npz', 'data/derived/schema.json',
        'data/derived/participants.csv', 'scripts/evaluate.py', 'scripts/prepare.py', 'scripts/train_selected.py',
        'src/diagramo/features.py', 'EXPERIMENT_PROTOCOL.md', 'requirements-lock.txt')]
    write(OUT / 'baseline_manifest.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(protected))})
    write(OUT / 'config.json', config)
    write(OUT / 'provenance.json', dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
          branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip(),
          status_before=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT).decode(),
          python=sys.version, versions={p: importlib.metadata.version(p) for p in ('shap', 'numpy', 'scipy', 'pandas', 'lightgbm', 'scikit-learn', 'matplotlib', 'joblib')},
          dependency_graph={'kinematic': 'diagramo.branches.kinematic -> diagramo.features',
                            'classifier': 'evaluate.build_pipeline -> train-only imputer/variance/scaler -> LightGBM',
                            'reused': ['saved outer-fold LightGBM pipelines', 'saved fold assignments', 'evaluate.classifiers/build_pipeline/metrics'],
                            'historical_deleted_modules_restored': False}))
    write(OUT / 'validation.json', dict(complete=False, stage='initialized'))
    return config
