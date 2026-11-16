"""Run experiments A--F, using only Kinematic features and fixed LightGBM.

Existing baseline artifacts are read-only. The only classifier fits are the
required task-excluded models and train-only cross-fitted importance models.
"""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from kinematic_xai_common import ROOT, OUT, SEED, REPEATS, METRICS, load_data, initialize, sha, write, factory, metrics
from diagramo.kinematic_xai import (feature_mapping, blocks, explain_pipeline, permute_block,
    median_task_removal, prediction_changes, train_only_task_importance)

EXP = OUT / 'exploratory'


def scores(y, prediction, probability):
    return {k: v for k, v in metrics(y, prediction, probability).items() if k in METRICS}


def audit_mapping(X, ids, schema):
    from prepare import TASKS, canonical
    from diagramo.branches.kinematic import extract
    assert schema['tasks'] == TASKS
    mapping = feature_mapping(schema['tasks'], schema['features_per_task'])
    source_checks = []
    for i in (0, len(ids) // 2, len(ids) - 1):
        path = ROOT / 'data/clean_data' / (ids[i] + '_merged.json')
        record = json.loads(path.read_text(encoding='utf-8'))
        by_task = {canonical(key): value['data'] for key, value in record.items() if canonical(key) in TASKS}
        values = []
        for task in TASKS:
            v, names = extract(by_task[task])
            assert names == schema['features_per_task']
            values.extend(v)
        np.testing.assert_array_equal(np.asarray(values, dtype=np.float32), X[i])
        source_checks.append(dict(participant_id=ids[i], path=path.relative_to(ROOT).as_posix(), sha256=sha(path), tasks_checked=14))
    mapping.to_csv(OUT / 'feature_mapping.csv', index=False)
    write(OUT / 'mapping_validation.json', dict(complete=True, source_checks=source_checks,
         task_order=TASKS, groups_per_task=mapping[mapping.task_index == 0].group.value_counts().to_dict()))
    return mapping


def load_baseline(f, X, y, tr, te, oof):
    path = ROOT / 'results/models/kinematic/LightGBM' / f'fold_{f}.joblib'
    reconstructed = not path.exists()
    if reconstructed:
        pipe = factory().fit(X[tr], y[tr])
        target = OUT / 'checkpoints/reconstructed' / f'fold_{f}.joblib'
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, target, compress=3)
        path = target
    else:
        pipe = joblib.load(path)
    # Independently reproduce only fitted preprocessing; do not refit the baseline classifier.
    reference = clone(pipe[:-1]).fit(X[tr], y[tr])
    np.testing.assert_array_equal(reference[0].statistics_, pipe[0].statistics_)
    np.testing.assert_array_equal(reference[1].get_support(), pipe[1].get_support())
    np.testing.assert_array_equal(reference[2].mean_, pipe[2].mean_)
    np.testing.assert_array_equal(reference[2].scale_, pipe[2].scale_)
    np.testing.assert_array_equal(reference.transform(X[te]), pipe[:-1].transform(X[te]))
    assert pipe[-1].get_params() == factory()[-1].get_params()
    assert pipe[-1].booster_.dump_model()['tree_info'][0]['tree_structure']['internal_count'] == len(tr)
    exp = explain_pipeline(pipe, X[te])
    np.testing.assert_array_equal(exp['prediction'], oof.prediction.to_numpy()[te])
    np.testing.assert_allclose(exp['probability'], oof.probability.to_numpy()[te], atol=1e-12, rtol=0)
    return pipe, exp, dict(checkpoint=path.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(path),
                           reconstructed=reconstructed, train_preprocessing_matches=True)


def run_fold(f, X, y, ids, mapping, oof, tr, te, signature):
    completed = EXP / 'folds' / f'fold_{f}.json'
    if completed.exists():
        assert json.loads(completed.read_text())['analysis_signature'] == signature
        print('REUSE completed outer fold', f, flush=True)
        return
    began = time.perf_counter()
    pipe, exp, model_meta = load_baseline(f, X, y, tr, te, oof)
    for folder in ('shap', 'permutation', 'ablation', 'folds'):
        (EXP / folder).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXP / 'shap' / f'fold_{f}.npz',
                        train_indices=tr, test_indices=te, ids=ids[te], **exp)
    baseline = scores(y[te], exp['prediction'], exp['probability'])
    permutation_rows, orders, probabilities, predictions = [], [], [], []
    for li, level in enumerate(('task', 'group')):
        for bi, (name, columns) in enumerate(blocks(mapping, level).items()):
            for repeat in range(REPEATS):
                seed = int(np.random.SeedSequence([SEED, f, li, bi, repeat]).generate_state(1)[0])
                order = np.random.default_rng(seed).permutation(len(te))
                perturbed = permute_block(X[te], columns, order)
                pred = pipe.predict(perturbed)
                prob = pipe.predict_proba(perturbed)[:, 1]
                performance = scores(y[te], pred, prob)
                permutation_rows.append(dict(fold=f, level=level, block=name, repeat=repeat, seed=seed,
                    **performance, **{'drop_' + k: baseline[k] - performance[k] for k in METRICS}))
                orders.append(order)
                probabilities.append(prob)
                predictions.append(pred)
    pd.DataFrame(permutation_rows).to_csv(EXP / 'permutation' / f'fold_{f}.csv', index=False)
    np.savez_compressed(EXP / 'permutation' / f'fold_{f}.npz', test_indices=te,
                        orders=orders, probability=probabilities, prediction=predictions)
    ablations, fold_results = [], []
    for task, columns in blocks(mapping, 'task').items():
        keep = np.setdiff1d(np.arange(X.shape[1]), columns)
        diagnostic = median_task_removal(X[te], columns, pipe[0].statistics_)
        pred = pipe.predict(diagnostic)
        prob = pipe.predict_proba(diagnostic)[:, 1]
        variants = [('test_time_train_median', pred, prob, 0., None)]
        train_began = time.perf_counter()
        retrained = factory().fit(X[tr][:, keep], y[tr])
        seconds = time.perf_counter() - train_began
        pred_r = retrained.predict(X[te][:, keep])
        prob_r = retrained.predict_proba(X[te][:, keep])[:, 1]
        path = OUT / 'checkpoints/loto' / f'fold_{f}' / (task + '.joblib')
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(retrained, path, compress=3)
        variants.append(('leave_one_task_out_refit', pred_r, prob_r, seconds, path))
        for method, prediction, probability, train_seconds, checkpoint in variants:
            performance = scores(y[te], prediction, probability)
            fold_results.append(dict(fold=f, task=task, method=method,
                n_train=len(tr), n_test=len(te), n_features=X.shape[1] if checkpoint is None else len(keep),
                **performance, **{'delta_' + k: performance[k] - baseline[k] for k in METRICS},
                train_seconds=train_seconds, checkpoint=None if checkpoint is None else checkpoint.relative_to(ROOT).as_posix(),
                checkpoint_sha256=None if checkpoint is None else sha(checkpoint)))
            changes = prediction_changes(y[te], exp['prediction'], prediction)
            for i, participant in enumerate(te):
                ablations.append(dict(fold=f, participant_id=ids[participant], label=int(y[participant]), task=task,
                     method=method, baseline_prediction=int(exp['prediction'][i]), baseline_probability=exp['probability'][i],
                     prediction=int(prediction[i]), probability=probability[i], probability_delta=probability[i] - exp['probability'][i], change=changes[i]))
    pd.DataFrame(ablations).to_csv(EXP / 'ablation' / f'fold_{f}.oof.csv', index=False)
    pd.DataFrame(fold_results).to_csv(EXP / 'ablation' / f'fold_{f}.csv', index=False)
    safe = OUT / 'train_only' / f'outer_fold_{f}'
    safe.mkdir(parents=True, exist_ok=True)
    inner_metadata = []
    def save_inner(inner, fit, validation, inner_pipe, inner_exp):
        np.savez_compressed(safe / f'inner_{inner}.npz', fit_ids=ids[tr][fit], validation_ids=ids[tr][validation], **inner_exp)
        path = OUT / 'checkpoints/train_only' / f'outer_{f}_inner_{inner}.joblib'
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(inner_pipe, path, compress=3)
        inner_metadata.append(dict(inner_fold=inner, fit_ids=ids[tr][fit].tolist(), validation_ids=ids[tr][validation].tolist(),
              checkpoint=path.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(path)))
    importance, phi, inner_folds = train_only_task_importance(X[tr], y[tr], ids[tr], mapping,
              factory, seed=SEED, forbidden_ids=ids[te], save_split=save_inner)
    importance.to_csv(safe / 'task_importance.csv', index=False)
    np.savez_compressed(safe / 'cross_fitted.npz', ids=ids[tr], phi=phi, inner_fold=inner_folds)
    write(safe / 'provenance.json', dict(outer_train_ids=ids[tr].tolist(), forbidden_outer_test_ids=ids[te].tolist(),
          input_X_sha256=__import__('hashlib').sha256(X[tr].tobytes()).hexdigest(),
          input_y_sha256=__import__('hashlib').sha256(y[tr].tobytes()).hexdigest(),
          inner_splits=inner_metadata, no_task_selection_performed=True))
    write(completed, dict(fold=f, analysis_signature=signature, train_ids=ids[tr].tolist(), test_ids=ids[te].tolist(),
           baseline=baseline, **model_meta, max_raw_additivity_error=exp['max_raw_additivity_error'],
           max_sigmoid_probability_error=exp['max_sigmoid_probability_error'], permutation_evaluations=len(permutation_rows),
           loto_fits=14, train_only_fits=3, seconds=time.perf_counter() - began))
    print(f'OUTER {f} DONE: 780 permutations, 14 LOTO fits, 3 train-only fits; {time.perf_counter()-began:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folds', nargs='+', type=int, default=list(range(6)))
    args = ap.parse_args()
    if not set(args.folds) <= set(range(6)):
        raise ValueError('Outer fold IDs must be 0..5')
    initialize()
    code = {p: sha(ROOT / p) for p in ('src/diagramo/kinematic_xai.py', 'scripts/kinematic_xai_common.py', 'scripts/analyze_kinematic_xai.py')}
    signature = __import__('hashlib').sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    if (OUT / 'analysis_code.json').exists():
        assert json.loads((OUT / 'analysis_code.json').read_text()) == code, 'Analysis code changed: use a separate output directory'
    write(OUT / 'analysis_code.json', code)
    X, y, ids, schema, assignments, oof, splits = load_data()
    mapping = audit_mapping(X, ids, schema)
    assignments.to_csv(OUT / 'fold_assignments.csv', index=False)
    EXP.mkdir(parents=True, exist_ok=True)
    oof.to_csv(EXP / 'baseline_oof.csv', index=False)
    with threadpool_limits(limits=4):
        for f in args.folds:
            tr, te = splits[f]
            run_fold(f, X, y, ids, mapping, oof, tr, te, signature)
    completed = sorted((EXP / 'folds').glob('*.json'))
    write(OUT / 'validation.json', dict(complete=False, stage='analysis_finished' if len(completed) == 6 else 'partial_analysis',
          completed_outer_folds=len(completed), required_outer_folds=6))


if __name__ == '__main__':
    main()
