"""Lossless raw-index inventories and bounded stroke-aware windows."""
import numpy as np
def windows(pre,patch_size=128):
    n=len(pre['time']);start=np.flatnonzero(pre['boundary']);end=np.r_[start[1:],n]
    rows=[]
    for seg,(a,b)in enumerate(zip(start,end)):
        for p in range(a,b,patch_size):
            q=min(p+patch_size,b)
            rows.append((p,q,seg,int(pre['run'][p]),int(pre['pen'][p]),float(pre['time'][p]),float(pre['time'][q-1])))
    inventory=np.asarray(rows,dtype=np.float64)
    assert int(np.sum(inventory[:,1]-inventory[:,0]))==n
    np.testing.assert_array_equal(inventory[1:,0],inventory[:-1,1])
    assert inventory[0,0]==0 and inventory[-1,1]==n
    return inventory
def padded(pre,inventory,patch_size=128):
    P=len(inventory);C=pre['values'].shape[1]
    x=np.zeros((P,patch_size,C),dtype='float32');index_x=np.zeros_like(x)
    length=(inventory[:,1]-inventory[:,0]).astype('int32')
    for i,(a,b,*_)in enumerate(inventory):
        a,b=int(a),int(b);x[i,:b-a]=pre['values'][a:b];index_x[i,:b-a]=pre['index_values'][a:b]
    return x,index_x,length
def groups(indices,max_length):
    """Return padded token indices grouped by contiguous identity, with full coverage."""
    indices=np.asarray(indices);cuts=np.r_[0,np.flatnonzero(indices[1:]!=indices[:-1])+1,len(indices)]
    rows=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        for p in range(a,b,max_length):rows.append(np.arange(p,min(p+max_length,b)))
    result=np.full((len(rows),max_length),-1,dtype='int64')
    for i,row in enumerate(rows):result[i,:len(row)]=row
    np.testing.assert_array_equal(result[result>=0],np.arange(len(indices)))
    return result

