import numpy as np
NAMES=['mean','std','median','min','max','q05','q25','q75','q95','iqr','mad','absdiff_mean','absdiff_std','absdiff_median','absdiff_q95']
def features(x):
 q=np.quantile(x,[.05,.25,.5,.75,.95],axis=0);d=np.abs(np.diff(x,axis=0))
 if not len(d):d=np.zeros((1,x.shape[1]))
 return np.stack([x.mean(0),x.std(0),q[2],x.min(0),x.max(0),q[0],q[1],q[3],q[4],q[3]-q[1],np.median(np.abs(x-q[2]),axis=0),d.mean(0),d.std(0),np.median(d,axis=0),np.quantile(d,.95,axis=0)]).T.ravel()
def length_features(recordings):
 n=np.array([len(x) for x in recordings]);return np.array([len(n),n.sum(),n.mean(),n.std(),n.min(),n.max()],float)
def participant_features(recordings):return np.mean([features(x) for x in recordings],axis=0)
