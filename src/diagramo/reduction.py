"""Train-only reduction bank: share identical fitted transforms across a fixed grid.

Never fit this bank before splitting participants. An inner CV split needs its own
bank, not a bank fitted on the outer training set. No held-out labels are accepted.
"""
import copy
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def reduction_grid():
    return ([dict(name=f'pca_{k}', method='pca', value=k) for k in (16,32,64,128)]
            + [dict(name=f'pca_var{int(v*100)}', method='pca', value=v) for v in (.90,.95,.99)]
            + [dict(name=f'anova_{k}', method='anova', value=k) for k in (32,64,128,256,512)]
            + [dict(name=f'extratrees_{k}', method='extratrees', value=k) for k in (32,64,128,256,512)])


class ReductionBank:
    """A fold-local preprocessing Pipeline plus fitted PCA and supervised selectors.

    PCA uses exact full SVD once. Prefixes equal full-SVD PCA at each requested k.
    SelectKBest and SelectFromModel likewise reuse scores from the SAME train rows.
    Components are not whitened or rescaled after PCA.
    """
    def __init__(self, seed=42):
        self.seed=seed

    def fit(self, X, y, participant_ids):
        ids=np.asarray(participant_ids).astype(str)
        if len(ids)!=len(X) or len(set(ids))!=len(ids) or len(y)!=len(X):
            raise ValueError('Exactly one training row per unique participant is required')
        self.fit_ids_=ids.tolist()
        self.raw_features_=X.shape[1]
        self.preprocessing_=make_pipeline(SimpleImputer(strategy='median'),VarianceThreshold(),StandardScaler())
        A=self.preprocessing_.fit_transform(X,y)
        self.variable_indices_=self.preprocessing_[1].get_support(indices=True)
        self.pca_=PCA(svd_solver='full',random_state=self.seed).fit(A)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.feature_selection')
            self.anova_=SelectKBest(f_classif,k='all').fit(A,y)
        forest=ExtraTreesClassifier(n_estimators=200,max_features='sqrt',min_samples_leaf=2,
                                    n_jobs=1,random_state=self.seed).fit(A,y)
        self.trees_=SelectFromModel(forest,prefit=True,threshold=-np.inf,
                                   max_features=min(512,A.shape[1])).fit(A,y)
        self.variance_=np.cumsum(self.pca_.explained_variance_ratio_)
        self.fitted_features_=A.shape[1]
        return self

    def dimension(self, spec):
        if spec['method']=='none': return self.fitted_features_
        if spec['method']=='pca':
            v=spec['value']
            if isinstance(v,float):
                # sklearn PCA uses side='right' for an explained-variance target.
                return min(int(np.searchsorted(self.variance_,v,side='right')+1),len(self.fit_ids_)-1,self.fitted_features_)
            if v>min(len(self.fit_ids_)-1,self.fitted_features_):
                raise ValueError('Requested PCA dimension exceeds training-fold rank')
            return v
        return min(spec['value'],self.fitted_features_)

    def selector(self,spec):
        selector=copy.copy(self.anova_ if spec['method']=='anova' else self.trees_)
        if spec['method']=='anova': selector.set_params(k=self.dimension(spec))
        else:
            selector.set_params(max_features=self.dimension(spec))
            # SelectFromModel caches the validated budget during fit. Reusing the
            # SAME fitted forest requires updating that budget on this copy too.
            selector.max_features_=self.dimension(spec)
        return selector

    def prepare(self,X):
        A=self.preprocessing_.transform(X)
        return A,self.pca_.transform(A)

    def project(self, prepared, spec):
        A,P=prepared
        if spec['method']=='none': return A
        if spec['method']=='pca': return P[:,:self.dimension(spec)]
        return self.selector(spec).transform(A)

    def metadata(self,spec):
        d=dict(n_features=self.dimension(spec),method=spec['method'],value=spec['value'])
        if spec['method']=='pca': d['explained_variance']=float(self.variance_[d['n_features']-1])
        if spec['method'] in ('anova','extratrees'):
            d['selected_original_indices']=self.variable_indices_[self.selector(spec).get_support()].tolist()
        return d

    def audit(self):
        return dict(fit_participants=self.fit_ids_,raw_features=self.raw_features_,
                    variable_features=self.fitted_features_,
                    pca_explained_variance_ratio=self.pca_.explained_variance_ratio_.tolist(),
                    transforms=['median_imputer','variance_threshold','standard_scaler','PCA_full_SVD',
                                'SelectKBest_f_classif','SelectFromModel_ExtraTrees200'])


def choose_candidate(rows):
    """Prespecified train-CV ranking, independent of any outer hold-out result."""
    return sorted(rows,key=lambda r:(-r['mean_f1_macro'],-r['mean_accuracy'],-r['mean_auc'],
                                     r['mean_features'],r['reducer'],r['classifier']))[0]
