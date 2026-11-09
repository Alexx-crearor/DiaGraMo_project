"""74 documented reconstruction features; the paper omits individual formulas.

Derivatives use seconds and original coordinate units, across all recorded points.
Pen lifts use contiguous pen_status == 0 runs. No cognitive scores enter features.
"""
from collections import OrderedDict
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

CHANNELS = ('x', 'y', 'pressure', 'tilt', 'azimuth', 'time', 'pen_status')

def resample(a, length):
    a = np.asarray(a, dtype=np.float64)
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(a)), a).astype('float32')

def entropy(a):
    h = np.histogram(a, bins=32)[0].astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())

def moments(a):
    if np.std(a) < 1e-10:
        return 0., 0.
    return float(skew(a, bias=False)), float(kurtosis(a, bias=False))

def kinematic(record):
    a = {k: np.asarray(record[k], dtype=np.float64) for k in CHANNELS}
    # Keep only strictly increasing timestamps for finite differences.
    keep = np.r_[True, np.diff(a['time']) > 0]
    a = {k: v[keep] for k, v in a.items()}
    t, x, y = a['time'], a['x'], a['y']
    if len(t) < 4:
        raise ValueError('Recording has fewer than four unique timestamps')
    dt = np.diff(t)
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    speed = np.hypot(vx, vy)
    acc = np.hypot(ax, ay)
    jerk = np.hypot(np.gradient(ax, t), np.gradient(ay, t))
    p, tilt, az = a['pressure'], a['tilt'], a['azimuth']
    duration = t[-1] - t[0]
    peaks = find_peaks(speed, prominence=max(np.std(speed) * .1, 1e-8))[0]
    air = a['pen_status'] == 0
    starts = np.flatnonzero(air & ~np.r_[False, air[:-1]])
    ends = np.flatnonzero(air & ~np.r_[air[1:], False])
    air_durations = [t[e] - t[s] for s, e in zip(starts, ends)]
    length = np.hypot(np.diff(x), np.diff(y)).sum()
    curvature = np.abs(vx * ay - vy * ax) / np.maximum(speed ** 3, 1e-8)
    f = OrderedDict()
    def add(prefix, names, values):
        for k, v in zip(names, values):
            f[prefix + '_' + k] = float(v)
    add('velocity', ['mean','std','min','max','median','p25','p75','rms'],
        [speed.mean(),speed.std(),speed.min(),speed.max(),np.median(speed),*np.percentile(speed,[25,75]),np.sqrt(np.mean(speed**2))])
    add('acceleration', ['mean','std','max','median','p95','rms'],
        [acc.mean(),acc.std(),acc.max(),np.median(acc),np.percentile(acc,95),np.sqrt(np.mean(acc**2))])
    add('jerk', ['mean','std','rms'], [jerk.mean(),jerk.std(),np.sqrt(np.mean(jerk**2))])
    add('peaks', ['count','per_second','mean_height'], [len(peaks),len(peaks)/duration,speed[peaks].mean() if len(peaks) else 0])
    add('lifts', ['count','air_fraction','mean_air_duration'], [len(starts),air.mean(),np.mean(air_durations) if air_durations else 0])
    add('pressure', ['mean','std','min','max','median','p25','p75','rms'],
        [p.mean(),p.std(),p.min(),p.max(),np.median(p),*np.percentile(p,[25,75]),np.sqrt(np.mean(p**2))])
    add('orientation', ['tilt_mean','tilt_std','azimuth_mean','azimuth_std'], [tilt.mean(),tilt.std(),az.mean(),az.std()])
    add('trajectory', ['duration','path_length','displacement','width','height','bbox_area','straightness','on_surface_fraction'],
        [duration,length,np.hypot(x[-1]-x[0],y[-1]-y[0]),np.ptp(x),np.ptp(y),np.ptp(x)*np.ptp(y),np.hypot(x[-1]-x[0],y[-1]-y[0])/max(length,1e-8),1-air.mean()])
    add('curvature', ['mean','std'], [curvature.mean(),curvature.std()])
    add('statistics', ['speed_entropy','pressure_entropy','speed_skew','speed_kurtosis','acc_skew','acc_kurtosis','pressure_skew','pressure_kurtosis','x_skew','y_skew'],
        [entropy(speed),entropy(p),*moments(speed),*moments(acc),*moments(p),moments(x)[0],moments(y)[0]])
    # 3 RMS values and 4 power-spectrum descriptors for each of 4 channels = 19.
    add('energy', ['x_rms','y_rms','pressure_rms'], [np.sqrt(np.mean(z**2)) for z in [x,y,p]])
    for name, z in [('x',x),('y',y),('speed',speed),('pressure',p)]:
        uniform = np.interp(np.linspace(t[0],t[-1],len(t)), t, z)
        power = np.abs(np.fft.rfft(uniform-uniform.mean()))**2
        freq = np.fft.rfftfreq(len(uniform), duration/(len(t)-1))
        power[0] = 0
        total = power.sum()
        q = power / total if total > 1e-15 else np.zeros_like(power)
        centroid = (q * freq).sum()
        add('fft_'+name, ['dominant_hz','centroid_hz','bandwidth_hz','entropy'],
            [freq[np.argmax(power)] if total>1e-15 else 0,centroid,np.sqrt((q*(freq-centroid)**2).sum()),-(q[q>0]*np.log2(q[q>0])).sum()])
    assert len(f) == 74, len(f)
    values = np.asarray(list(f.values()),dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite kinematic features')
    return values, list(f)

def gaf(signals):
    """Gramian angular summation field; per-record, per-channel [-1,1] scaling."""
    lo, hi = signals.min(-1,keepdims=True), signals.max(-1,keepdims=True)
    z = np.divide(2*(signals-lo), hi-lo, out=np.ones_like(signals), where=(hi-lo)>1e-8)-1
    z = np.clip(z,-1,1)
    s = np.sqrt(np.maximum(0,1-z*z))
    return z[..., :,None]*z[...,None,:] - s[..., :,None]*s[...,None,:]

def cwt(signals):
    """Morlet, 224 log-spaced scales 1..128, 512 samples, resize time to 224.

    Log magnitude and per-image [0,1] scaling are explicit reconstruction choices.
    """
    import pywt
    z=(signals-signals.mean(-1,keepdims=True))/np.maximum(signals.std(-1,keepdims=True),1e-8)
    coef,_=pywt.cwt(z,np.geomspace(1,128,224),'morl',method='fft',axis=-1)
    image=np.log1p(np.abs(coef)).transpose(1,0,2)
    positions=np.linspace(0,signals.shape[-1]-1,224)
    left=np.floor(positions).astype(int); right=np.minimum(left+1,signals.shape[-1]-1); alpha=positions-left
    image=(image[...,left]*(1-alpha)+image[...,right]*alpha).astype('float32')
    image/=np.maximum(image.max(axis=(-2,-1),keepdims=True),1e-8)
    return image
