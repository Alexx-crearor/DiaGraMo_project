"""Small unordered task-token networks and explicit missing-task masks.

Task identity embeddings are optional labels, never temporal positions.
Attention is returned for supplementary inspection, not as causal attribution.
"""
import random
import os
import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def deterministic_seed(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TaskPreprocessor:
    """Train-only median/scaling, retaining all 74 coordinates in every token."""
    def fit(self, X):
        X = np.asarray(X).reshape(len(X), 1036)
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(X)
        self.scaler = StandardScaler().fit(self.imputer.transform(X))
        self.n_fit = len(X)
        return self

    def transform(self, X):
        X = np.asarray(X).reshape(len(X), 1036)
        result = self.scaler.transform(self.imputer.transform(X)).reshape(-1, 14, 74).astype('float32')
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite standardized task features')
        return result


class TaskBlock(nn.Module):
    def __init__(self, dim, heads, dropout, interactions=True):
        super().__init__()
        self.interactions, self.heads = interactions, heads
        self.norm_attention = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3) if interactions else None
        self.out_attention = nn.Linear(dim, dim) if interactions else None
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, present):
        attention = None
        if self.interactions:
            batch, tokens, dim = x.shape
            qkv = self.qkv(self.norm_attention(x)).reshape(batch, tokens, 3, self.heads, dim // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            score = (q @ k.transpose(-2, -1)) / (dim // self.heads) ** .5
            score = score.masked_fill(~present[:, None, None, :], float('-inf'))
            attention = score.softmax(dim=-1)
            context = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
            x = x + self.dropout(self.out_attention(context))
        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x.masked_fill(~present[:, :, None], 0.), attention


class KinematicTaskNet(nn.Module):
    def __init__(self, d_model=32, heads=2, layers=1, dropout=.1, pooling='mean', task_type=False, interactions=True):
        super().__init__()
        self.config = dict(d_model=d_model, heads=heads, layers=layers, dropout=dropout,
                           pooling=pooling, task_type=task_type, interactions=interactions)
        self.project = nn.Sequential(nn.Linear(74, d_model), nn.GELU(), nn.Dropout(dropout))
        self.task_type = nn.Embedding(14, d_model) if task_type else None
        self.blocks = nn.ModuleList([TaskBlock(d_model, heads, dropout, interactions) for _ in range(layers)])
        self.pool_score = nn.Linear(d_model, 1) if pooling == 'attention' else None
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, present=None, task_ids=None, return_attention=False):
        if present is None:
            present = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if torch.any(~present.any(dim=1)):
            raise ValueError('At least one task must be present for every participant')
        if task_ids is None:
            task_ids = torch.arange(x.shape[1], device=x.device)
        # Zero absent input before projection: missing values cannot affect another token.
        x = self.project(x.masked_fill(~present[:, :, None], 0.))
        if self.task_type is not None:
            x = x + self.task_type(task_ids)[None, :, :]
        x = x.masked_fill(~present[:, :, None], 0.)
        attentions = []
        for block in self.blocks:
            x, attention = block(x, present)
            if attention is not None:
                attentions.append(attention)
        if self.pool_score is None:
            weights = present.float() / present.sum(dim=1, keepdim=True)
        else:
            score = self.pool_score(x).squeeze(-1).masked_fill(~present, float('-inf'))
            weights = score.softmax(dim=1)
        pooled = (x * weights[:, :, None]).sum(dim=1)
        logits = self.head(self.final_norm(pooled)).squeeze(-1)
        return (logits, attentions, weights) if return_attention else logits


def architecture(candidate, family):
    return dict(d_model=candidate['d_model'], heads=candidate['heads'], layers=candidate['layers'],
                dropout=candidate['dropout'], pooling='mean' if family == 'task_mlp' else candidate['pooling'],
                task_type=candidate['task_type'], interactions=family != 'task_mlp')


def integrated_gradients(model, X, present, steps=64, device='cpu'):
    """Gauss-Legendre IG of the positive-class logit from Train-mean (zero) input."""
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    present = np.asarray(present, dtype=bool)
    if present.ndim == 1:
        present = np.broadcast_to(present, X.shape[:2]).copy()
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    nodes, weights = (nodes + 1) / 2, weights / 2
    values = np.zeros_like(X, dtype=np.float64)
    # Chunk alphas and participants to bound memory independently of steps.
    for i in range(0, len(X), 16):
        batch = torch.tensor(X[i:i+16], device=device)
        mask = torch.tensor(present[i:i+16], device=device)
        integrated = torch.zeros_like(batch, dtype=torch.float64)
        for j in range(0, steps, 8):
            alpha = torch.tensor(nodes[j:j+8], dtype=batch.dtype, device=device)
            point = (alpha[:, None, None, None] * batch[None]).reshape(-1, 14, 74).detach().requires_grad_(True)
            masks = mask.repeat(len(alpha), 1)
            gradient = torch.autograd.grad(model(point, masks).sum(), point)[0].reshape(len(alpha), *batch.shape)
            w = torch.tensor(weights[j:j+8], dtype=torch.float64, device=device)
            integrated += (gradient.double() * w[:, None, None, None]).sum(dim=0)
        values[i:i+len(batch)] = (batch.double() * integrated).detach().cpu().numpy()
    values *= present[:, :, None]
    with torch.no_grad():
        x = torch.tensor(X, device=device)
        mask = torch.tensor(present, device=device)
        logits = model(x, mask).cpu().numpy()
        reference = model(torch.zeros_like(x), mask).cpu().numpy()
    return values, reference, logits, values.sum(axis=(1, 2)) - (logits - reference)
