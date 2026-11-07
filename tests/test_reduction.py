"""Check cached fold transforms against independently fitted sklearn estimators."""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from diagramo.reduction import ReductionBank


class ReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.X = rng.normal(size=(80, 65))
        cls.X[:, -1] = 7  # Removed using training data only.
        cls.X[0, 0] = np.nan
        cls.y = np.tile([0, 1], 40)
        cls.ids = np.array([f'p{i}' for i in range(80)])
        cls.bank = ReductionBank().fit(cls.X[:60], cls.y[:60], cls.ids[:60])

    def test_pca_matches_independent_full_svd(self):
        train = self.bank.preprocessing_.transform(self.X[:60])
        test = self.bank.preprocessing_.transform(self.X[60:])
        prepared = self.bank.prepare(self.X[60:])
        for value in (16, 32, .90, .95, .99):
            spec = dict(method='pca', value=value)
            reference = PCA(n_components=value, svd_solver='full').fit(train)
            actual = self.bank.project(prepared, spec)
            np.testing.assert_allclose(actual, reference.transform(test), atol=1e-10)

    def test_selectors_match_independent_budgets(self):
        train = self.bank.preprocessing_.transform(self.X[:60])
        test = self.bank.preprocessing_.transform(self.X[60:])
        prepared = self.bank.prepare(self.X[60:])
        # Decrease and then increase budgets to catch stale selector caches.
        for k in (32, 8, 64, 16):
            references = {
                'anova': SelectKBest(f_classif, k=k).fit(train, self.y[:60]),
                'extratrees': SelectFromModel(
                    self.bank.trees_.estimator_, prefit=True,
                    threshold=-np.inf, max_features=k).fit(train, self.y[:60]),
            }
            for method, reference in references.items():
                actual = self.bank.project(prepared, dict(method=method, value=k))
                self.assertEqual(actual.shape[1], k)
                np.testing.assert_array_equal(actual, reference.transform(test))

    def test_held_out_transform_cannot_change_fitted_state(self):
        before = (self.bank.preprocessing_[0].statistics_.copy(),
                  self.bank.preprocessing_[2].mean_.copy(),
                  self.bank.pca_.components_.copy(),
                  self.bank.anova_.scores_.copy())
        self.bank.prepare(np.full((20, 65), 1e9))
        after = (self.bank.preprocessing_[0].statistics_,
                 self.bank.preprocessing_[2].mean_,
                 self.bank.pca_.components_, self.bank.anova_.scores_)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(self.bank.fit_ids_, self.ids[:60].tolist())

    def test_duplicate_participants_rejected(self):
        with self.assertRaises(ValueError):
            ReductionBank().fit(self.X[:60], self.y[:60], ['same'] * 60)


if __name__ == '__main__':
    unittest.main()
