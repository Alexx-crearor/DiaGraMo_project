"""Label-free raw-record perturbations; all features use the original extractor.

These are controlled sensitivity hypotheses, not proven label-preserving changes.
"""
import numpy as np
from diagramo.features import CHANNELS


def motor_normalize(record):
    a = {k: np.asarray(record[k], dtype=np.float64).copy() for k in CHANNELS}
    xy = np.stack([a['x'] - a['x'][0], a['y'] - a['y'][0]])
    scale = np.sqrt(np.mean(np.sum(xy * xy, axis=0)))
    scale = scale if scale > 1e-12 else 1.
    a['x'], a['y'] = xy / scale
    p_scale = np.quantile(np.abs(a['pressure']), .95)
    a['pressure'] /= p_scale if p_scale > 1e-12 else 1.
    # Keep actual seconds, pen state, tilt and azimuth; do not erase duration.
    return a


def perturb_record(record, kind, training=False):
    a = {k: np.asarray(record[k], dtype=np.float64).copy() for k in CHANNELS}
    magnitude = .05 if training else .10
    if kind == 'translation':
        a['x'] += magnitude * max(np.ptp(a['x']), 1.)
        a['y'] -= magnitude * max(np.ptp(a['y']), 1.)
    elif kind == 'spatial_scale':
        for key in ('x', 'y'):
            a[key] = a[key][0] + (a[key] - a[key][0]) * (1 + magnitude)
    elif kind == 'pressure_scale':
        a['pressure'] *= 1 + magnitude
    elif kind == 'sampling_half':
        keep = np.unique(np.r_[np.arange(0, len(a['time']), 2), len(a['time']) - 1])
        if len(keep) < 4:
            raise ValueError('Sampling perturbation needs at least four retained samples')
        a = {k: v[keep] for k, v in a.items()}
    elif kind == 'timing_warp':
        duration = a['time'][-1] - a['time'][0]
        amplitude = .01 if training else .02
        if duration <= 0:
            raise ValueError('Timing perturbation requires positive recorded duration')
        phase = (a['time'] - a['time'][0]) / duration
        a['time'] += amplitude * duration * np.sin(2 * np.pi * phase)
    elif kind == 'duration_scale':
        a['time'] = a['time'][0] + (a['time'] - a['time'][0]) * (1 + magnitude)
    else:
        raise ValueError(kind)
    if not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError('Nonfinite transformed raw signal')
    return a
