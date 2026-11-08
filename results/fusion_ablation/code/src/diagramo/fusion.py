"""Small train-only branch transforms and cross-fitted probability fusion.

No function in this module needs outer-test labels. Candidate selection is kept
separate from fitting; callers must provide genuinely held-out probabilities.
"""
import copy
import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BranchBank:
    """Share preprocessing/PCA only among candidates using identical train IDs."""
    def fit(self, X, y, ids, specs):
        ids = np.asarray(ids).astype(str)
        if len(ids) != len(X) or len(y) != len(X) or len(set(ids)) != len(ids):
            raise ValueError('One row per unique training participant required')
        self.fit_ids_ = ids.tolist()
        self.raw_features_ = X.shape[1]
        self.pre_ = make_pipeline(SimpleImputer(strategy='median'), VarianceThreshold(), StandardScaler())
        A = self.pre_.fit_transform(X, y)
        self.variable_ = self.pre_[1].get_support(indices=True)
        self.pca_ = None
        self.anova_ = None
        if any(s.startswith('pca') for s in specs):
            self.pca_ = PCA(svd_solver='full', random_state=42).fit(A)
        if any(s.startswith('anova') for s in specs):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                self.anova_ = SelectKBest(f_classif, k='all').fit(A, y)
        self.metadata_ = {}
        for spec in specs:
            if spec == 'none':
                d = dict(n_features=A.shape[1])
            elif spec.startswith('pca_var'):
                target = int(spec.removeprefix('pca_var')) / 100
                k = min(int(np.searchsorted(np.cumsum(self.pca_.explained_variance_ratio_), target,
                                            side='right') + 1), len(ids)-1, A.shape[1])
                d = dict(n_features=k, explained_variance=float(self.pca_.explained_variance_ratio_[:k].sum()))
            elif spec.startswith('anova_'):
                k = min(int(spec.split('_')[1]), A.shape[1])
                sel = copy.copy(self.anova_).set_params(k=k)
                d = dict(n_features=k, selected_original_indices=self.variable_[sel.get_support()].tolist())
            else:
                raise ValueError('Unsupported reducer: '+spec)
            self.metadata_[spec] = d
        return self

    def transform_many(self, X):
        A = self.pre_.transform(X)
        P = self.pca_.transform(A) if self.pca_ is not None else None
        result = {}
        for spec, meta in self.metadata_.items():
            k = meta['n_features']
            if spec == 'none': result[spec] = A
            elif spec.startswith('pca'): result[spec] = P[:, :k]
            else: result[spec] = copy.copy(self.anova_).set_params(k=k).transform(A)
        return result

    def audit(self):
        return dict(fit_participants=self.fit_ids_, raw_features=self.raw_features_,
                    reducers=self.metadata_, preprocessing=['median','variance','standard_scaler'])


def check_crossfit(ids, splits):
    """Reject overlap, missing rows, duplicate holdouts or an altered cohort."""
    ids = np.asarray(ids).astype(str)
    if len(set(ids)) != len(ids): raise ValueError('Duplicate participant')
    coverage = np.zeros(len(ids), dtype=int)
    for tr, va in splits:
        tr, va = np.asarray(tr), np.asarray(va)
        if set(tr) & set(va) or set(tr) | set(va) != set(range(len(ids))):
            raise ValueError('Invalid cross-fitting partition')
        coverage[va] += 1
    if not np.all(coverage == 1): raise ValueError('Cross-fitting coverage is not exactly one')


class ProbabilityFusion:
    """Convex pooling or uncertainty-gated auxiliary correction of column 0.

Fit exclusively on cross-fitted probabilities. Adaptive correction is bounded
by budget <= .5 and disabled outside the chosen Kinematic uncertainty region.
Neither confidence gating nor convex constraints guarantee better test scores.
"""
    def __init__(self, kind='late', ridge=0.1, budget=0.5, gate=0.25):
        self.kind, self.ridge, self.budget, self.gate = kind, ridge, budget, gate

    def fit(self, P, y):
        P = np.asarray(P, dtype=float)
        y = np.asarray(y)
        if P.ndim != 2 or P.shape[1] < 2 or len(P) != len(y): raise ValueError('Invalid probability matrix')
        if not np.isfinite(P).all() or (P < 0).any() or (P > 1).any(): raise ValueError('Invalid probabilities')
        if self.kind not in ('equal','late','adaptive','kinematic'): raise ValueError('Invalid fusion kind')
        if not 0 <= self.budget <= .5 or not 0 <= self.gate <= .5: raise ValueError('Invalid adaptive limits')
        n = P.shape[1] - (self.kind == 'adaptive')
        self.weights_ = np.full(n, 1/n)
        self.optimizer_success_ = True
        if self.kind in ('late','adaptive'):
            def objective(w):
                p = np.clip(self._pool(P, w), 1e-8, 1-1e-8)
                return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)) + self.ridge*np.sum((w-1/n)**2))
            result = minimize(objective, self.weights_, method='SLSQP', bounds=[(0,1)]*n,
                              constraints=[dict(type='eq',fun=lambda w: w.sum()-1)],
                              options=dict(maxiter=200, ftol=1e-10))
            if not result.success: raise RuntimeError('Fusion optimization failed: '+result.message)
            self.weights_ = np.maximum(result.x, 0)
            self.weights_ /= self.weights_.sum()
        return self

    def _pool(self, P, w):
        if self.kind == 'kinematic': return P[:,0]
        if self.kind == 'adaptive':
            g = (np.abs(P[:,0]-.5) <= self.gate).astype(float)
            return P[:,0] + g*self.budget*(P[:,1:]@w-P[:,0])
        return P@w

    def predict_proba(self, P):
        p = self._pool(np.asarray(P), self.weights_)
        return np.column_stack([1-p,p])

    def metadata(self):
        return dict(kind=self.kind, ridge=self.ridge, budget=self.budget, gate=self.gate,
                    weights=self.weights_.tolist(), optimizer_success=self.optimizer_success_)


def choose_inner(rows, baseline=None):
    """Only inner-CV aggregate scores enter this function.

    Optional selective safety rule requires strictly better inner F1 and no
    decrease in inner Accuracy; otherwise it returns the Kinematic candidate.
    """
    if not rows: raise ValueError('Empty candidate pool')
    eligible = rows
    if baseline is not None:
        eligible = [r for r in rows if r['mean_f1_macro'] > baseline['mean_f1_macro']+1e-12
                    and r['mean_accuracy'] >= baseline['mean_accuracy']-1e-12]
        if not eligible: return baseline
    return sorted(eligible, key=lambda r: (-r['mean_f1_macro'], -r['mean_accuracy'],
                  -r['mean_auc'], r['mean_features'], r['candidate']))[0]
