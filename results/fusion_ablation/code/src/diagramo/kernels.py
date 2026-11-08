"""Exact RBF kernel precomputation to avoid repeated high-dimensional dot products."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

class RBFKernelTransformer(TransformerMixin,BaseEstimator):
    def __init__(self,gamma='scale'): self.gamma=gamma
    def fit(self,X,y=None):
        self.X_fit_=np.asarray(X,dtype=np.float64)
        if self.gamma=='scale':
            var=self.X_fit_.var(); self.gamma_=1/(self.X_fit_.shape[1]*var) if var else 1.
        elif self.gamma=='auto': self.gamma_=1/self.X_fit_.shape[1]
        else: self.gamma_=float(self.gamma)
        return self
    def transform(self,X):
        return rbf_kernel(np.asarray(X,dtype=np.float64),self.X_fit_,gamma=self.gamma_)
