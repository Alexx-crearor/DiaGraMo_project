"""Per-person sufficient statistics; combining them is restricted to Train IDs."""
from raw_temporal_common import *
from diagramo.temporal.dataset import load_person,sufficient_statistics

def main():
    ids,y,schema,folds,splits=cohort();rows=[];comparisons=[]
    for i,pid in enumerate(ids):
        p=load_person(str(OUT/'cache'),pid)
        # Statistics always use original float32 cache, not compact RAM copies.
        totals=np.zeros((4,21),dtype='float64');count=0
        with np.load(OUT/'cache'/(pid+'.npz'))as raw:
            for ti in range(14):
                n=raw[f'n{ti}'];mask=np.arange(128)[None,:]<n[:,None];count+=int(n.sum())
                for j,key in enumerate(('x','i')):
                    A=raw[f'{key}{ti}'][mask].astype('float64');totals[j*2]+=A.sum(0);totals[j*2+1]+=np.square(A).sum(0)
        rows.append(totals/count)
        for ti,inv in enumerate(p['inventories']):
            sizes=pd.DataFrame(dict(segment=inv[:,2],n=inv[:,1]-inv[:,0])).groupby('segment').n.sum().to_numpy()
            for size in (64,128,256):comparisons.append(dict(participant_id=pid,task=schema['tasks'][ti],patch_size=size,
                patches=int(np.ceil(sizes/size).sum()),raw_samples=int(sizes.sum()),padded_cells=int(np.ceil(sizes/size).sum()*size-sizes.sum())))
        if i%30==0:print('STATISTICS',i+1,flush=True)
    np.savez_compressed(OUT/'cache/statistics.npz',ids=ids,statistics=rows)
    pd.DataFrame(comparisons).to_csv(OUT/'audit/window_tradeoff.csv',index=False)
    write(OUT/'cache/statistics.json',dict(complete=True,sha256=sha(OUT/'cache/statistics.npz'),fit_from_labels=False,
       note='Store per-participant first/second moments; TrainNormalizer combines only supplied fit participants with equal participant weighting',
       normalization='signed log1p continuous input; Train participant-balanced mean/SD; fixed SD floor0.1; saturation +/-10 z; flags unscaled'))
    print('STATISTICS COMPLETE',flush=True)

if __name__=='__main__':main()
