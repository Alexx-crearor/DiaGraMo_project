"""Shared read-only half-storage bank; computations and gradients stay float32."""
from raw_temporal_common import *

def main():
    ids,y,schema,_,_=cohort();inventory=pd.read_csv(OUT/'audit/prepared_inventory.csv')
    total=int(inventory.patches.sum());target=OUT/'cache/packed.npy'
    bank=np.lib.format.open_memmap(target,mode='w+',dtype='float16',shape=(2,total,128,21))
    offsets={};start=0;maxerror=0.
    for i,pid in enumerate(ids):
        n=int(inventory[inventory.participant_id==pid].patches.sum());offsets[pid]=[start,start+n]
        with np.load(OUT/'cache'/(pid+'.npz'))as a:
            p=start
            for ti in range(14):
                x=a[f'x{ti}'];ix=a[f'i{ti}'];k=len(x)
                for j,A in enumerate((x,ix)):
                    q=A.astype('float16');maxerror=max(maxerror,float(np.max(np.abs(A-q.astype('float32')))));bank[j,p:p+k]=q
                p+=k
        start+=n
        if i%30==0:print('PACKED',i+1,flush=True)
    bank.flush();del bank
    write(OUT/'cache/packed.json',dict(complete=True,offsets=offsets,shape=[2,total,128,21],dtype='float16',maximum_absolute_log_channel_quantization_error=maxerror,
       arithmetic='float32, not AMP',sha256=sha(target)))
    print('PACKING COMPLETE; max input quantization error',maxerror,flush=True)

if __name__=='__main__':main()
