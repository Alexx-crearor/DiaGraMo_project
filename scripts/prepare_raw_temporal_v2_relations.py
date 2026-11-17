"""Unlabeled stroke relationship metadata from existing caches, no SVC reread."""
from raw_temporal_v2_common import *
from diagramo.temporal.dataset import load_person

DIRECTORY=LOCAL/'relationships'


def prepare(pid):
    item=load_person(str(CACHE),pid);x=item['x'].astype('float32');length=item['length'];segment=item['patch_segment']
    mask=np.arange(x.shape[1])[None,:]<length[:,None];count=len(item['segment_task'])
    # Undo signed-log for approximate source-device coordinates. The packed
    # cache is quantized; these are not asserted to be exact physical positions.
    xy=np.sign(x[:,:,:2])*np.expm1(np.abs(x[:,:,:2]));xy*=mask[:,:,None]
    sums=np.zeros((count,2));numbers=np.zeros(count);pen=np.zeros(count)
    np.add.at(sums,segment,xy.sum(1));np.add.at(numbers,segment,length)
    np.add.at(pen,segment,(x[:,:,7]*mask).sum(1))
    times=[]
    for inventory in item['inventories']:
        times.append(inventory[:,5:7]-inventory[0,5])
    times=np.concatenate(times);first=np.r_[0,np.flatnonzero(np.diff(segment))+1];last=np.r_[first[1:]-1,len(segment)-1]
    start=times[first,0];end=times[last,1];ordinal=np.empty(count)
    for task in range(14):
        at=np.flatnonzero(item['segment_task']==task);ordinal[at]=np.arange(len(at))
    result=np.column_stack([start,end,end-start,sums/numbers[:,None],pen/numbers,ordinal]).astype('float32')
    assert result.shape==(count,7) and np.isfinite(result).all() and (result[:,2]>=0).all()
    return result


if __name__=='__main__':
    DIRECTORY.mkdir(parents=True,exist_ok=True);ids,y,schema,_,_=cohort();rows=[]
    sig=signature('configs/raw_temporal_v2/phase1.json',['scripts/prepare_raw_temporal_v2_relations.py'])
    for index,pid in enumerate(ids):
        path=DIRECTORY/f'{pid}.npy';meta=path.with_suffix('.json')
        if path.exists():
            saved=json.loads(meta.read_text());assert saved['signature']==sig['signature'] and sha(path)==saved['sha256']
        else:
            values=prepare(pid);np.save(path,values)
            saved=dict(participant_id=pid,sha256=sha(path),signature=sig['signature'],segments=len(values));write(meta,saved)
        rows.append(saved)
        if index%25==0:print('RELATIONSHIPS',index+1,flush=True)
    write(OUT/'phase2_temporal/relationship_manifest.json',dict(complete=True,signature=sig,records=rows,
        columns=['start_seconds_within_task','end_seconds_within_task','duration_seconds','approx_center_x_device_units',
            'approx_center_y_device_units','pen_down_fraction','stroke_ordinal_within_task'],
        no_label_fitting=True,source='Existing original packed input cache and float64 source-time inventories',
        precision='Coordinates inverted from float16 signed-log cache; approximate device units, not mm or exact sensor values'))
