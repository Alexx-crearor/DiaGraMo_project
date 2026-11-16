"""Independent replay of Kinematic SHAP, every permutation and every ablation.

Use --numerics-only before executing the viewer notebook. Final validation also
checks figure provenance, notebook outputs, tests and frozen numerical hashes.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from kinematic_xai_common import ROOT, OUT, SEED, METRICS, REPEATS, sha, write, factory, load_data, metrics
from diagramo.kinematic_xai import (feature_mapping, blocks, aggregate, check_additivity,
    explain_pipeline, permute_block, median_task_removal, prediction_changes)

EXP = OUT / 'exploratory'
TABLES = EXP / 'tables'


def close(a, b, atol=1e-12):
    np.testing.assert_allclose(a, b, atol=atol, rtol=0, equal_nan=False)


def protected():
    baseline = json.loads((OUT / 'baseline_manifest.json').read_text())
    for rel, digest in baseline.items():
        assert sha(ROOT / rel) == digest, 'Baseline changed: ' + rel
    for rel, digest in json.loads((OUT / 'analysis_code.json').read_text()).items():
        assert sha(ROOT / rel) == digest, 'Analysis code changed: ' + rel
    return len(baseline)


def train_preprocessing(pipe, X, y):
    check = clone(pipe[:-1]).fit(X, y)
    np.testing.assert_array_equal(check[0].statistics_, pipe[0].statistics_)
    np.testing.assert_array_equal(check[1].get_support(), pipe[1].get_support())
    np.testing.assert_array_equal(check[2].mean_, pipe[2].mean_)
    np.testing.assert_array_equal(check[2].scale_, pipe[2].scale_)
    assert pipe[-1].get_params() == factory()[-1].get_params()
    assert pipe[-1].booster_.dump_model()['tree_info'][0]['tree_structure']['internal_count'] == len(X)


def numerical_files():
    paths = [OUT / 'feature_mapping.csv', OUT / 'fold_assignments.csv', OUT / 'config.json', OUT / 'mapping_validation.json']
    for folder in ('exploratory', 'train_only'):
        paths.extend(p for p in (OUT / folder).rglob('*') if p.is_file())
    paths.extend(p for p in (OUT / 'checkpoints').rglob('*.joblib'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def numerics():
    preserved = protected()
    X, y, ids, schema, assignments, oof, splits = load_data()
    mapping = pd.read_csv(OUT / 'feature_mapping.csv')
    pd.testing.assert_frame_equal(mapping, feature_mapping(schema['tasks'], schema['features_per_task']))
    source = json.loads((OUT / 'mapping_validation.json').read_text())
    for record in source['source_checks']:
        assert sha(ROOT / record['path']) == record['sha256']
    signature_code = json.loads((OUT / 'analysis_code.json').read_text())
    signature = hashlib.sha256(json.dumps(signature_code, sort_keys=True).encode()).hexdigest()
    all_phi = np.zeros_like(X, dtype=np.float64)
    max_raw, max_probability = 0., 0.
    permutation_count, loto_count, safe_count, reconstructed = 0, 0, 0, 0
    ablation_rows = []
    for f, (tr, te) in enumerate(splits):
        meta = json.loads((EXP / 'folds' / f'fold_{f}.json').read_text())
        assert meta['analysis_signature'] == signature
        assert meta['train_ids'] == ids[tr].tolist() and meta['test_ids'] == ids[te].tolist()
        assert not set(meta['train_ids']) & set(meta['test_ids'])
        assert sha(ROOT / meta['checkpoint']) == meta['checkpoint_sha256']
        reconstructed += int(meta['reconstructed'])
        pipe = joblib.load(ROOT / meta['checkpoint'])
        train_preprocessing(pipe, X[tr], y[tr])
        actual = explain_pipeline(pipe, X[te])
        with np.load(EXP / 'shap' / f'fold_{f}.npz') as saved:
            np.testing.assert_array_equal(saved['train_indices'], tr)
            np.testing.assert_array_equal(saved['test_indices'], te)
            np.testing.assert_array_equal(saved['ids'], ids[te])
            for key in ('phi', 'base', 'raw_score', 'probability', 'prediction', 'kept_columns'):
                close(saved[key], actual[key])
            all_phi[te] = saved['phi']
        close(actual['probability'], oof.probability.to_numpy()[te])
        np.testing.assert_array_equal(actual['prediction'], oof.prediction.to_numpy()[te])
        baseline = metrics(y[te], actual['prediction'], actual['probability'])
        for k in METRICS:
            close(meta['baseline'][k], baseline[k])
        max_raw = max(max_raw, actual['max_raw_additivity_error'])
        max_probability = max(max_probability, actual['max_sigmoid_probability_error'])
        perm = pd.read_csv(EXP / 'permutation' / f'fold_{f}.csv', float_precision='round_trip')
        assert len(perm) == 26 * REPEATS
        levels = {level: list(blocks(mapping, level)) for level in ('task', 'group')}
        with np.load(EXP / 'permutation' / f'fold_{f}.npz') as d:
            np.testing.assert_array_equal(d['test_indices'], te)
            for i, row in enumerate(perm.itertuples()):
                li = ('task', 'group').index(row.level)
                bi = levels[row.level].index(row.block)
                seed = int(np.random.SeedSequence([SEED, f, li, bi, row.repeat]).generate_state(1)[0])
                assert seed == row.seed
                order = np.random.default_rng(seed).permutation(len(te))
                np.testing.assert_array_equal(order, d['orders'][i])
                test = permute_block(X[te], blocks(mapping, row.level)[row.block], order)
                prob, pred = pipe.predict_proba(test)[:, 1], pipe.predict(test)
                close(prob, d['probability'][i])
                np.testing.assert_array_equal(pred, d['prediction'][i])
                measured = metrics(y[te], pred, prob)
                for k in METRICS:
                    close(getattr(row, k), measured[k])
                    close(getattr(row, 'drop_' + k), baseline[k] - measured[k])
                permutation_count += 1
        ablation = pd.read_csv(EXP / 'ablation' / f'fold_{f}.oof.csv', float_precision='round_trip')
        afold = pd.read_csv(EXP / 'ablation' / f'fold_{f}.csv', float_precision='round_trip')
        assert len(ablation) == 14 * 2 * len(te) and len(afold) == 28
        for task, columns in blocks(mapping, 'task').items():
            keep = np.setdiff1d(np.arange(1036), columns)
            assert len(columns) == 74 and len(keep) == 962 and not set(columns) & set(keep)
            for method in ('test_time_train_median', 'leave_one_task_out_refit'):
                table_row = afold[(afold.method == method) & (afold.task == task)].iloc[0]
                d = ablation[(ablation.method == method) & (ablation.task == task)]
                np.testing.assert_array_equal(d.participant_id, ids[te])
                np.testing.assert_array_equal(d.label, y[te])
                if method == 'test_time_train_median':
                    test = median_task_removal(X[te], columns, pipe[0].statistics_)
                    pred, prob = pipe.predict(test), pipe.predict_proba(test)[:, 1]
                else:
                    assert sha(ROOT / table_row.checkpoint) == table_row.checkpoint_sha256
                    model = joblib.load(ROOT / table_row.checkpoint)
                    assert model.n_features_in_ == 962
                    train_preprocessing(model, X[tr][:, keep], y[tr])
                    pred, prob = model.predict(X[te][:, keep]), model.predict_proba(X[te][:, keep])[:, 1]
                    loto_count += 1
                np.testing.assert_array_equal(pred, d.prediction)
                np.testing.assert_array_equal(actual['prediction'], d.baseline_prediction)
                close(prob, d.probability)
                close(d.probability_delta, prob - actual['probability'])
                close(d.baseline_probability, actual['probability'])
                np.testing.assert_array_equal(prediction_changes(y[te], actual['prediction'], pred), d.change)
                for k, value in metrics(y[te], pred, prob).items():
                    if k in METRICS:
                        close(table_row[k], value)
                        close(table_row['delta_' + k], value - baseline[k])
        ablation_rows.append(ablation)
        safe = OUT / 'train_only' / f'outer_fold_{f}'
        info = json.loads((safe / 'provenance.json').read_text())
        assert info['outer_train_ids'] == ids[tr].tolist()
        assert info['forbidden_outer_test_ids'] == ids[te].tolist()
        assert info['input_X_sha256'] == hashlib.sha256(X[tr].tobytes()).hexdigest()
        assert info['input_y_sha256'] == hashlib.sha256(y[tr].tobytes()).hexdigest()
        cross = np.load(safe / 'cross_fitted.npz')
        np.testing.assert_array_equal(cross['ids'], ids[tr])
        safe_phi, coverage = np.zeros((len(tr), 1036)), np.zeros(len(tr), dtype=int)
        expected = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X[tr], y[tr]))
        for item, (fit, validation) in zip(info['inner_splits'], expected):
            inner = item['inner_fold']
            assert item['fit_ids'] == ids[tr][fit].tolist()
            assert item['validation_ids'] == ids[tr][validation].tolist()
            assert not set(item['fit_ids']) & set(item['validation_ids'])
            assert not set(ids[te]) & (set(item['fit_ids']) | set(item['validation_ids']))
            assert sha(ROOT / item['checkpoint']) == item['checkpoint_sha256']
            model = joblib.load(ROOT / item['checkpoint'])
            train_preprocessing(model, X[tr][fit], y[tr][fit])
            exp = explain_pipeline(model, X[tr][validation])
            d = np.load(safe / f'inner_{inner}.npz')
            np.testing.assert_array_equal(d['fit_ids'], ids[tr][fit])
            np.testing.assert_array_equal(d['validation_ids'], ids[tr][validation])
            for key in ('phi', 'base', 'raw_score', 'probability', 'prediction'):
                close(d[key], exp[key])
            safe_phi[validation] = d['phi']
            coverage[validation] += 1
            assert (cross['inner_fold'][validation] == inner).all()
            safe_count += 1
        assert np.all(coverage == 1)
        close(safe_phi, cross['phi'])
        safe_rank = pd.read_csv(safe / 'task_importance.csv').set_index('task')
        masses = aggregate(safe_phi, mapping, 'task').mean()
        close(safe_rank.loc[masses.index].mean_absolute_shap_mass, masses)
        close(safe_rank.loc[masses.index]['rank'], rankdata(-masses))
        print(f'VALIDATED outer {f}: SHAP, 780 permutations, 14 refits, 3 train-only models', flush=True)
    # Independently check aggregate tables, including ranks and signed cancellation.
    for level in ('feature', 'task', 'group'):
        by = {r.full_name: np.array([r.column]) for r in mapping.itertuples()} if level == 'feature' else blocks(mapping, level)
        summary = pd.read_csv(TABLES / (level + '_importance.csv')).set_index('full_name' if level == 'feature' else level)
        names = list(by)
        means = np.array([np.abs(all_phi[:, by[name]]).sum(axis=1).mean() for name in names])
        global_ranks = rankdata(-means)
        fold_means = np.array([[np.abs(all_phi[te][:, by[name]]).sum(axis=1).mean() for name in names] for _, te in splits])
        ranks = np.array([rankdata(-row) for row in fold_means])
        for i, name in enumerate(names):
            row = summary.loc[name]
            close(row.mean_absolute_shap_mass, means[i])
            close(row.total_absolute_shap_mass, means[i] * 258)
            close(row.mass_per_feature, means[i] / len(by[name]))
            close(row.share_percent, means[i] / means.sum() * 100)
            close(row.global_rank, global_ranks[i])
            close(row.mean_rank, ranks[:, i].mean())
            close(row.rank_sd, ranks[:, i].std(ddof=1))
            close(row.top3_count, (ranks[:, i] <= 3).sum())
        per_fold = pd.read_csv(TABLES / (level + '_importance_by_fold.csv'))
        for f in range(6):
            frame = per_fold[per_fold.fold == f].set_index(level)
            close(frame.loc[names].mean_absolute_shap_mass, fold_means[f])
            close(frame.loc[names]['rank'], ranks[f])
    participants = pd.read_csv(TABLES / 'participant_task_attributions.csv')
    for i, pid in enumerate(ids):
        d = participants[participants.participant_id == pid].set_index('task')
        signed, mass = [], []
        for task, columns in blocks(mapping, 'task').items():
            close(d.loc[task, 'absolute_shap_mass'], np.abs(all_phi[i, columns]).sum())
            close(d.loc[task, 'signed_shap_log_odds'], all_phi[i, columns].sum())
            signed.append(d.loc[task, 'signed_shap_log_odds'])
            mass.append(d.loc[task, 'absolute_shap_mass'])
        close(np.sum(signed), all_phi[i].sum())
        close(np.sum(mass), np.abs(all_phi[i]).sum())
    combined = pd.concat(ablation_rows, ignore_index=True)
    summary = pd.read_csv(TABLES / 'task_ablation_summary.csv')
    full = pd.read_csv(TABLES / 'task_ablation_oof.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(full, combined)
    for (method, task), d in combined.groupby(['method', 'task']):
        assert len(d) == 258 and d.participant_id.is_unique
        r = summary[(summary.method == method) & (summary.task == task)].iloc[0]
        for k, value in metrics(d.label, d.prediction, d.probability).items():
            if k in METRICS:
                close(r[k], value)
                baseline = metrics(y, oof.prediction, oof.probability)[k]
                close(r['delta_' + k], value - baseline)
        for change in ('corrected', 'harmed', 'unchanged_correct', 'unchanged_wrong'):
            close(r[change], (d.change == change).sum())
        close(r.corrected - r.harmed, (d.prediction == d.label).sum() - (oof.prediction == oof.label).sum())
    permutation = pd.read_csv(TABLES / 'permutation_repeats.csv')
    psum = pd.read_csv(TABLES / 'permutation_importance.csv').set_index(['level', 'block'])
    for key, d in permutation.groupby(['level', 'block']):
        assert len(d) == 180
        for k in METRICS:
            per_fold = d.groupby('fold')['drop_' + k].mean()
            close(psum.loc[key, 'drop_' + k + '_mean'], per_fold.mean())
            close(psum.loc[key, 'drop_' + k + '_std'], per_fold.std(ddof=1))
    assert permutation_count == 4680 and loto_count == 84 and safe_count == 18
    result = dict(complete=True, checks=dict(protected_baseline_files=preserved, outer_folds=6,
          participants=258, features=1036, tasks=14, source_groups=len(blocks(mapping, 'group')),
          baseline_oof_matches=True, baseline_checkpoints_reconstructed=reconstructed,
          raw_additivity_passed=True, max_raw_additivity_error=max_raw, max_sigmoid_probability_error=max_probability,
          permutation_evaluations_replayed=permutation_count, loto_models_validated=loto_count,
          ablation_prediction_rows=len(combined), train_only_models_validated=safe_count,
          train_only_id_separation_passed=True, all_task_and_group_aggregates_checked=True,
          paired_prediction_changes_checked=True), numerical_sha256=numerical_files(),
          numeric_code_sha256={p: sha(ROOT / p) for p in ('scripts/report_kinematic_xai.py', 'scripts/validate_kinematic_xai.py')})
    write(OUT / 'numeric_validation.json', result)
    print('NUMERICAL VALIDATION PASSED', flush=True)


def finalize():
    import nbformat
    from PIL import Image
    numeric = json.loads((OUT / 'numeric_validation.json').read_text())
    assert numeric['complete']
    preserved = protected()
    for rel, digest in {**numeric['numerical_sha256'], **numeric['numeric_code_sha256']}.items():
        assert sha(ROOT / rel) == digest, 'Frozen numeric artifact/code changed: ' + rel
    manifest = json.loads((OUT / 'figures/manifest.json').read_text())
    assert len(manifest) == 16
    for row in manifest:
        for rel, digest in row['sources'].items():
            assert sha(ROOT / rel) == digest, rel
        for ext, digest in row['outputs'].items():
            p = OUT / 'figures' / (row['name'] + '.' + ext)
            assert sha(p) == digest and p.stat().st_size > 1000
            if ext == 'png':
                with Image.open(p) as im:
                    assert min(im.info['dpi']) >= 299
                    assert im.width == round(row['width_inches'] * 300)
    nb_path = ROOT / 'notebooks/05_kinematic_task_xai.ipynb'
    n = nbformat.read(nb_path, as_version=4)
    nbformat.validate(n)
    code_cells = [c for c in n.cells if c.cell_type == 'code']
    assert code_cells and all(c.execution_count is not None for c in code_cells)
    assert not [o for c in code_cells for o in c.outputs if o.output_type == 'error']
    numeric_tables = sum('text/html' in o.get('data', {}) for c in code_cells for o in c.outputs)
    assert numeric_tables >= 15
    notebooks = json.loads((OUT / 'notebook_validation.json').read_text())
    assert len(notebooks) == 1 and notebooks[0]['notebook'] == nb_path.name and notebooks[0]['errors'] == 0
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_kinematic_xai.py', '-v'],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT / 'tests.log').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    assert tests.returncode == 0, tests.stderr
    dependency = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, encoding='utf-8')
    assert dependency.returncode == 0, dependency.stdout + dependency.stderr
    source_paths = [ROOT / p for p in ('src/diagramo/kinematic_xai.py', 'tests/test_kinematic_xai.py', 'requirements-kinematic-xai.txt')]
    source_paths += list((ROOT / 'scripts').glob('*kinematic_xai*.py'))
    for path in source_paths:
        if path.suffix == '.py':
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    write(OUT / 'validation.json', dict(complete=True, stage='complete', **numeric['checks'],
          protected_baseline_files_final=preserved, figure_pairs=len(manifest), figure_dpi=300,
          notebook_executed=True, executed_code_cells=len(code_cells), numeric_table_outputs=numeric_tables,
          unit_tests=8, dependency_check=dependency.stdout.strip(), numerical_validation_sha256=sha(OUT / 'numeric_validation.json'),
          notebook_sha256=sha(nb_path), report_sha256=sha(OUT / 'REPORT_VI.md'),
          code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(source_paths))},
          transformer_implemented=False, other_representation_experiments_run=False,
          outer_test_labels_used_for_task_selection=False, xai_is_exploratory=True))
    print('FINAL VALIDATION COMPLETE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--numerics-only', action='store_true')
    args = ap.parse_args()
    with threadpool_limits(limits=4):
        numerics() if args.numerics_only else finalize()
