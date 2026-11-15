"""Snapshot and verify original reproduction artifacts without fitting models.

Run --snapshot before cleanup, then run without arguments after cleanup.
The inventory includes local ignored data/caches; it never deletes files.
"""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/cleanup'
EXPERIMENTS = ('feature_optimization', 'embedding_ablation', 'fusion_ablation', 'xai')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def snapshot():
    if (OUT / 'baseline_manifest.json').exists():
        raise FileExistsError('The pre-cleanup manifest already exists')
    original = set(git('ls-tree', '-r', '--name-only', 'ac4f927').splitlines())
    tracked = set(git('ls-files').splitlines())
    removable = sorted(p for p in tracked - original
                       if p.startswith(('scripts/', 'tests/', 'src/', 'notebooks/'))
                       or p == 'requirements-xai.txt')
    removable += sorted(p for p in tracked if any(p.startswith('results/' + e + '/') for e in EXPERIMENTS))
    # Check every Python file, including later experiments, before choosing removals.
    dependencies = {}
    for rel in sorted(p for p in tracked if p.endswith('.py')):
        tree = ast.parse((ROOT / rel).read_text(encoding='utf-8-sig'))
        dependencies[rel] = sorted({n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
                                   | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
    baseline_python = [p for p in original if p.endswith('.py')]
    removed_modules = {Path(p).stem for p in removable if p.endswith('.py') and p.startswith(('scripts/', 'src/'))}
    for rel in baseline_python:
        assert not any(d.split('.')[-1] in removed_modules for d in dependencies[rel]), rel
    rows = []
    for rel in sorted(tracked | {'DiaGraMo_REVIEW_KEEP_DECISIONS_VI.md'}):
        category = 'IMPROVEMENT' if rel in removable else 'BASELINE' if rel in original else 'UNKNOWN'
        if rel in ('README.md', '.gitignore', 'scripts/evaluate.py', 'scripts/extract.py',
                   'scripts/execute_notebooks.py', 'src/diagramo/features.py', 'src/diagramo/kernels.py'):
            category = 'SHARED'
        rows.append({'path': rel, 'category': category,
                     'decision': 'remove' if rel in removable else 'preserve; unknown user file' if category == 'UNKNOWN' else 'keep/review'})
    ignored = []
    for e in EXPERIMENTS:
        folder = ROOT / 'results' / e
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.relative_to(ROOT).as_posix() not in tracked:
                ignored.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                                'decision': 'remove: experiment-only directory'})
    write('origin.json', {'pre_cleanup_sha': git('rev-parse', 'HEAD'),
                          'branch': git('branch', '--show-current'), 'baseline_commit': 'ac4f927',
                          'untracked_before': git('ls-files', '--others', '--exclude-standard').splitlines()})
    write('classification.json', rows)
    write('dependencies_before.json', dependencies)
    write('removal_plan.json', {'tracked': removable, 'ignored_experiment_files': ignored,
                               'directories': ['results/' + e for e in EXPERIMENTS]})
    # Preserve every baseline output, all notebooks, all data, and downloaded weights.
    paths = set()
    for folder in ('results', 'data', '.cache', 'references'):
        for p in (ROOT / folder).rglob('*'):
            if p.is_file() and not any(p.is_relative_to(ROOT / 'results' / e) for e in EXPERIMENTS):
                paths.add(p)
    paths.update(ROOT / p for p in original if p.startswith('notebooks/'))
    paths.update(ROOT / p for p in ('requirements-lock.txt', 'EXPERIMENT_PROTOCOL.md'))
    manifest = []
    for i, p in enumerate(sorted(paths)):
        manifest.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
        if i % 1000 == 0:
            print('Hashed', i + 1, '/', len(paths), flush=True)
    write('baseline_manifest.json', manifest)
    print('SNAPSHOT', len(manifest), 'protected files;', len(removable), 'tracked removals;', len(ignored), 'ignored removals')


def verify():
    manifest = json.loads((OUT / 'baseline_manifest.json').read_text(encoding='utf-8'))
    for row in manifest:
        p = ROOT / row['path']
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from evaluate import classifiers, metrics
    folds = pd.read_csv(ROOT / 'results/cv/fold_assignments.csv')
    splits = list(StratifiedKFold(6, shuffle=True, random_state=42).split(folds.participant_id, folds.label))
    assert len(folds) == 258 and folds.participant_id.is_unique
    for f, (_, te) in enumerate(splits):
        assert (folds.iloc[te].fold == f).all()
    summary = pd.read_csv(ROOT / 'results/cv/summary.csv')
    assert len(summary) == 168 and summary.representation.nunique() == 21
    assert all(set(g.classifier) == set(classifiers(42)) for _, g in summary.groupby('representation'))
    oofs = sorted((ROOT / 'results/cv').glob('*.oof.csv'))
    assert len(oofs) == 168
    for path in oofs:
        d = pd.read_csv(path, float_precision='round_trip')
        assert d[['participant_id', 'label', 'fold']].equals(folds)
        assert d.participant_id.is_unique and d.prediction.isin([0, 1]).all() and d.probability.between(0, 1).all()
        r = json.loads(path.with_name(path.name.replace('.oof.csv', '.json')).read_text())
        selected = summary[(summary.representation == r['representation']) & (summary.classifier == r['classifier'])]
        assert len(selected) == 1 and len(r['folds']) == 6
        for key, value in metrics(d.label, d.prediction, d.probability).items():
            assert np.isclose(value, r['pooled'][key], atol=1e-12, rtol=0)
            assert np.isclose(value, selected.iloc[0][key], atol=1e-12, rtol=0)
        for f in range(6):
            part = d[d.fold == f]
            saved = next(x for x in r['folds'] if x['fold'] == f)
            for key, value in metrics(part.label, part.prediction, part.probability).items():
                assert np.isclose(value, saved[key], atol=1e-12, rtol=0)
    validation = json.loads((ROOT / 'results/validation.json').read_text())
    assert validation['complete'] and validation['actual_results'] == validation['oof_files_validated'] == 168
    plan = json.loads((OUT / 'removal_plan.json').read_text())
    assert all(not (ROOT / p).exists() for p in plan['tracked'])
    assert all(not (ROOT / p).exists() for p in plan['directories'])
    write('artifact_validation.json', {'complete': True, 'protected_files_unchanged': len(manifest),
          'protected_bytes': sum(r['bytes'] for r in manifest), 'cv_results': 168, 'oof_files': 168,
          'participants': 258, 'folds': 6, 'seed': 42, 'all_pooled_and_fold_metrics_match': True})
    print('VERIFIED', len(manifest), 'unchanged protected files; 168/168 CV and OOF, all metrics')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', action='store_true')
    args = ap.parse_args()
    snapshot() if args.snapshot else verify()
