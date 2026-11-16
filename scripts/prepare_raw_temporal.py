"""Cache deterministic raw windows; no cohort-fitted normalization or model fitting."""
import concurrent.futures,time
from raw_temporal_common import *
from diagramo.temporal.preprocessing import preprocess,CHANNELS
from diagramo.temporal.segmentation import windows,padded,groups

def job(item):
    pid,tasks,patch=item;target=OUT/'cache'/(pid+'.npz');records,meta,path=recording_map(pid,tasks)
    if target.exists():return json.loads((OUT/'cache'/(pid+'.json')).read_text())
    arrays={};inventories=[];offset=0;seg_offset=0;ns=0
    for ti,task in enumerate(tasks):
        pre=preprocess(records[task][1]);inv=windows(pre,patch);x,index_x,length=padded(pre,inv,patch)
        arrays[f'x{ti}']=x;arrays[f'i{ti}']=index_x;arrays[f'n{ti}']=length;arrays[f'm{ti}']=inv
        inventories.append(dict(task=task,raw_samples=len(pre['time']),patches=len(inv),segments=int(pre['segment'].max()+1),
           sample_coverage=int(length.sum()),duration=float(pre['time'][-1]-pre['time'][0]),max_abs_derived=float(np.abs(pre['physical'][:,8:13]).max()),
           valid_velocity=int(pre['velocity_valid'].sum()),valid_acceleration=int(pre['acceleration_valid'].sum()),
           total_padded_cells=int(x.shape[0]*patch),padding_cells=int(x.shape[0]*patch-length.sum())))
    target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**arrays)
    row=dict(participant_id=pid,source_sha256=sha(path),raw_source_sha256={task:sha(raw_svc_path(pid,records[task][0]))for task in tasks},cache_sha256=sha(target),tasks=inventories)
    write(target.with_suffix('.json'),row);return row

def main():
    ids,y,schema,folds,splits=cohort();config=json.loads((ROOT/'configs/raw_temporal.json').read_text());dest=OUT/'cache';dest.mkdir(exist_ok=True)
    code={p:sha(ROOT/p)for p in ('src/diagramo/temporal/preprocessing.py','src/diagramo/temporal/segmentation.py','scripts/prepare_raw_temporal.py','configs/raw_temporal.json')}
    if (dest/'code.json').exists():assert json.loads((dest/'code.json').read_text())==code
    else:write(dest/'code.json',code)
    began=time.perf_counter();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3)as pool:
        for i,row in enumerate(pool.map(job,[(pid,schema['tasks'],config['patch_size'])for pid in ids])):
            rows.append(row)
            if i%20==0:print('PREPARED',i+1,'/258',flush=True)
    rowsflat=[dict(participant_id=row['participant_id'],**task)for row in rows for task in row['tasks']]
    pd.DataFrame(rowsflat).to_csv(OUT/'audit/prepared_inventory.csv',index=False)
    write(dest/'manifest.json',rows);write(dest/'validation.json',dict(complete=True,participants=258,recordings=len(rowsflat),
         raw_samples=sum(r['raw_samples']for r in rowsflat),covered_samples=sum(r['sample_coverage']for r in rowsflat),
         channels=list(CHANNELS),patch_size=config['patch_size'],seconds=time.perf_counter()-began,normalization_fit=False))
    print('PREPARATION COMPLETE',time.perf_counter()-began,flush=True)

if __name__=='__main__':main()
