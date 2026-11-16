"""Kinematic-only attribution utilities for the current four-branch repository.

Raw TreeSHAP describes a frozen binary LightGBM's log-odds, not probability.
Absolute feature sums are attribution mass, not grouped Shapley values.
"""
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold


def feature_mapping(tasks, feature_names):
    if len(tasks) != 14 or len(set(tasks)) != 14 or len(feature_names) != 74 or len(set(feature_names)) != 74:
        raise ValueError('Expected 14 unique tasks and 74 unique source feature names')
    rows = []
    for ti, task in enumerate(tasks):
        for fi, name in enumerate(feature_names):
            group = name.split('_', 1)[0]
            rows.append(dict(column=ti * 74 + fi, task_index=ti, task=task,
                             feature_in_task=fi, feature=name, group=group,
                             full_name=task + '::' + name))
    return pd.DataFrame(rows)


def blocks(mapping, level):
    return {name: g.column.to_numpy(dtype=int) for name, g in mapping.groupby(level, sort=False)}


def aggregate(values, mapping, level, absolute=True):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(mapping):
        raise ValueError('Attributions must be sample by original feature column')
    return pd.DataFrame({name: (np.abs(values[:, cols]) if absolute else values[:, cols]).sum(axis=1)
                         for name, cols in blocks(mapping, level).items()})


def check_additivity(phi, base, raw, probability, atol=1e-9):
    phi, base, raw, probability = map(np.asarray, (phi, base, raw, probability))
    if not all(np.isfinite(a).all() for a in (phi, base, raw, probability)):
        raise ValueError('Nonfinite model or SHAP output')
    reconstruction_error = float(np.max(np.abs(phi.sum(axis=1) + base - raw)))
    probability_error = float(np.max(np.abs(expit(raw) - probability)))
    if reconstruction_error > atol or probability_error > atol:
        raise ValueError(f'Additivity/output-space failure: raw={reconstruction_error}, probability={probability_error}')
    return dict(max_raw_additivity_error=reconstruction_error, max_sigmoid_probability_error=probability_error)


def explain_pipeline(pipe, X):
    """Explain only supplied evaluation rows; reference counts come from Train trees.

    Median imputation, variance filtering and scaling are already fitted in pipe.
    Dropped constant columns receive zero; remaining coordinates retain order.
    """
    X = np.asarray(X)
    transformed = pipe[:-1].transform(X)
    mask = pipe.named_steps['variancethreshold'].get_support()
    if len(mask) != X.shape[1] or transformed.shape[1] != mask.sum():
        raise ValueError('Unexpected preprocessing schema')
    model = pipe[-1]
    explainer = shap.TreeExplainer(model, model_output='raw', feature_perturbation='tree_path_dependent')
    exp = explainer(transformed, check_additivity=True)
    if exp.values.shape != transformed.shape:
        raise ValueError('Expected one binary-class raw-score attribution matrix')
    phi = np.zeros(X.shape, dtype=np.float64)
    phi[:, mask] = exp.values
    base = np.broadcast_to(np.asarray(exp.base_values).reshape(-1), (len(X),)).copy()
    raw = np.asarray(model.predict(transformed, raw_score=True))
    probability = pipe.predict_proba(X)[:, 1]
    errors = check_additivity(phi, base, raw, probability)
    return dict(phi=phi, base=base, raw_score=raw, probability=probability,
                prediction=pipe.predict(X), kept_columns=np.flatnonzero(mask), **errors)


def permute_block(X, columns, order):
    order = np.asarray(order)
    if not np.array_equal(np.sort(order), np.arange(len(X))):
        raise ValueError('Expected a permutation of all evaluation rows')
    result = np.array(X, copy=True)
    result[:, columns] = X[order][:, columns]
    return result


def median_task_removal(X, columns, training_medians):
    """Diagnostic replacement only; preserves the original fitted model dimensions."""
    if len(training_medians) != X.shape[1] or not np.isfinite(training_medians).all():
        raise ValueError('Expected finite medians fitted on training data')
    result = np.array(X, copy=True)
    result[:, columns] = np.asarray(training_medians)[columns]
    return result


def prediction_changes(y, baseline_prediction, prediction):
    before, after = np.asarray(baseline_prediction) == y, np.asarray(prediction) == y
    return np.select([~before & after, before & ~after, before & after],
                     ['corrected', 'harmed', 'unchanged_correct'], default='unchanged_wrong')


def train_only_task_importance(X_train, y_train, train_ids, mapping, pipeline_factory, seed=42,
                               forbidden_ids=(), save_split=None):
    """Cross-fit importance strictly inside the supplied training cohort.

    Safe use inside a later inner-CV search: pass that inner-training subset,
    never inner-validation or outer-test rows. No task subset is selected here.
    save_split receives each fitted pipeline and inner-validation explanations.
    """
    X_train, y_train, train_ids = np.asarray(X_train), np.asarray(y_train), np.asarray(train_ids)
    if len(X_train) != len(y_train) or len(y_train) != len(train_ids) or len(set(train_ids)) != len(train_ids):
        raise ValueError('Training rows, labels and unique IDs must align')
    if set(train_ids) & set(forbidden_ids):
        raise ValueError('Held-out IDs entered train-only attribution')
    phi = np.zeros(X_train.shape, dtype=np.float64)
    validation_fold = np.full(len(y_train), -1)
    seen = np.zeros(len(y_train), dtype=int)
    for f, (fit, validation) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(X_train, y_train)):
        pipe = pipeline_factory()
        pipe.fit(X_train[fit], y_train[fit])
        exp = explain_pipeline(pipe, X_train[validation])
        phi[validation] = exp['phi']
        validation_fold[validation] = f
        seen[validation] += 1
        if save_split:
            save_split(f, fit, validation, pipe, exp)
    if not np.all(seen == 1):
        raise ValueError('Each training participant must be explained exactly once out of inner training')
    importance = aggregate(phi, mapping, 'task').mean().rename('mean_absolute_shap_mass').to_frame()
    importance.index.name = 'task'
    importance['rank'] = importance.mean_absolute_shap_mass.rank(ascending=False, method='average')
    return importance.reset_index(), phi, validation_fold
