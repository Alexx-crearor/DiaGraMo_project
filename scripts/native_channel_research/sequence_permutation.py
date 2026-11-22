import hashlib
import numpy as np
def permute_rows(x,key,seed=42):
 if len(x)<2:return x.copy()
 n=int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:8],'little')
 at=np.random.default_rng(n).permutation(len(x))
 if np.array_equal(at,np.arange(len(x))):at=np.roll(at,1)
 return x[at].copy()
