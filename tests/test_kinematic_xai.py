"""Independent semantic, mapping, perturbation and leakage regression tests."""
import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from diagramo.kinematic_xai import (feature_mapping, aggregate, blocks, permute_block,
    median_task_removal, check_additivity, explain_pipeline, train_only_task_importance, prediction_changes)
from evaluate import build_pipeline


class KinematicXAITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / 'data/derived/schema.json').read_text())
        cls.mapping = feature_mapping(cls.schema['tasks'], cls.schema['features_per_task'])

    def test_source_mapping_and_groups(self):
        from prepare import TASKS
        self.assertEqual(list(blocks(self.mapping, 'task')), TASKS)
        self.assertEqual(self.mapping.iloc[0].full_name, TASKS[0] + '::velocity_mean')
        self.assertEqual(self.mapping.iloc[-1].full_name, 'copy::fft_pressure_entropy')
        self.assertEqual(set(self.mapping.groupby('task').size()), {74})
        self.assertEqual(len(blocks(self.mapping, 'group')['fft']), 14 * 16)
        self.assertEqual(len(blocks(self.mapping, 'group')), 12)
        with self.assertRaises(ValueError):
            feature_mapping(['copy'] * 14, self.schema['features_per_task'])

    def test_absolute_mass_is_not_absolute_net_or_group_shapley(self):
        phi = np.zeros((2, 1036))
        phi[:, :2] = [2., -2.]
        phi[1, 74] = 3.
        mass = aggregate(phi, self.mapping, 'task')
        signed = aggregate(phi, self.mapping, 'task', absolute=False)
        self.assertEqual(mass.iloc[0, 0], 4.)
        self.assertEqual(signed.iloc[0, 0], 0.)
        np.testing.assert_array_equal(mass.sum(axis=1), np.abs(phi).sum(axis=1))
        np.testing.assert_array_equal(signed.sum(axis=1), phi.sum(axis=1))

    def test_grouped_permutation_uses_shared_rows(self):
        X = np.arange(4 * 6).reshape(4, 6)
        order = np.array([2, 0, 3, 1])
        result = permute_block(X, [1, 3], order)
        np.testing.assert_array_equal(result[:, [1, 3]], X[order][:, [1, 3]])
        np.testing.assert_array_equal(result[:, [0, 2, 4, 5]], X[:, [0, 2, 4, 5]])
        np.testing.assert_array_equal(X, np.arange(24).reshape(4, 6))
        with self.assertRaises(ValueError):
            permute_block(X, [1, 3], [0, 0, 2, 3])

    def test_test_time_removal_is_train_reference_with_original_dimensions(self):
        X = np.array([[100., 50., 3.], [900., 80., 8.]])
        result = median_task_removal(X, [0, 1], np.array([2., 4., 7.]))
        np.testing.assert_array_equal(result, [[2., 4., 3.], [2., 4., 8.]])
        self.assertEqual(result.shape, X.shape)
        np.testing.assert_array_equal(X[:, 0], [100., 900.])

    def test_log_odds_and_probability_spaces_are_distinct(self):
        phi, base, raw = np.array([[.2, .3]]), np.array([-.2]), np.array([.3])
        check_additivity(phi, base, raw, expit(raw))
        with self.assertRaises(ValueError):
            check_additivity(phi, base, expit(raw), expit(raw))
        with self.assertRaises(ValueError):
            check_additivity(phi + .01, base, raw, expit(raw))

    def test_alignment_of_constant_columns_on_actual_saved_model(self):
        d = np.load(ROOT / 'data/derived/kinematic.npz')
        pipe = joblib.load(ROOT / 'results/models/kinematic/LightGBM/fold_0.joblib')
        folds = pd.read_csv(ROOT / 'results/cv/fold_assignments.csv')
        held_out = np.flatnonzero(folds.fold.to_numpy() == 0)[:3]
        exp = explain_pipeline(pipe, d['X'][held_out])
        dropped = ~pipe[1].get_support()
        self.assertTrue(dropped.any())
        self.assertTrue((exp['phi'][:, dropped] == 0).all())
        self.assertLess(exp['max_raw_additivity_error'], 1e-9)
        transformed = pipe[:-1].transform(d['X'][held_out])
        native = pipe[-1].booster_.predict(transformed, pred_contrib=True)
        np.testing.assert_allclose(exp['phi'][:, ~dropped], native[:, :-1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(exp['base'], native[:, -1], atol=1e-12, rtol=0)

    def test_train_only_cross_fitting_and_forbidden_ids(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(30, 1036)).astype('float32')
        y = np.tile([0, 1], 15)
        ids = np.array(['train_' + str(i) for i in range(30)])
        records = []
        def factory():
            return build_pipeline('LightGBM', LGBMClassifier(n_estimators=4, num_leaves=3, min_child_samples=2,
                                  verbosity=-1, n_jobs=1, random_state=42))
        def receive(f, fit, validation, pipe, exp):
            self.assertFalse(set(ids[fit]) & set(ids[validation]))
            self.assertEqual(len(fit), 20)
            self.assertEqual(len(validation), 10)
            records.extend(validation.tolist())
        with threadpool_limits(limits=1):
            importance, phi, folds = train_only_task_importance(X, y, ids, self.mapping, factory,
                    forbidden_ids=['outer_test_only'], save_split=receive)
        self.assertEqual(sorted(records), list(range(30)))
        self.assertEqual(phi.shape, X.shape)
        self.assertEqual(set(folds), {0, 1, 2})
        self.assertEqual(len(importance), 14)
        with self.assertRaisesRegex(ValueError, 'Held-out IDs'):
            train_only_task_importance(X, y, ids, self.mapping, factory, forbidden_ids=[ids[0]])

    def test_paired_changes_are_not_accuracy_only(self):
        y, old, new = np.array([0, 1, 0, 1]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0])
        change = prediction_changes(y, old, new)
        self.assertEqual(change.tolist(), ['corrected', 'harmed', 'unchanged_correct', 'unchanged_wrong'])
        self.assertEqual(int((change == 'corrected').sum() - (change == 'harmed').sum()), int((new == y).sum() - (old == y).sum()))


if __name__ == '__main__':
    unittest.main()
