"""Trace merged raw sequences back to original SVC exports, without altering input."""
import concurrent.futures
from raw_temporal_common import *
ORDER=('x','y','time','pen_status','azimuth','tilt','pressure')
def job(item):
    pid,tasks=item;records,meta,path=recording_map(pid,tasks,raw_svc=False);rows=[]
    for task in tasks:
        key,record=records[task];source=ROOT/'data/raw_data/handwriting_data/data_SVC'/pid/(key[:-5]+'.svc')
        row=dict(participant_id=pid,task=task,merged_source=path.relative_to(ROOT).as_posix(),raw_source=source.relative_to(ROOT).as_posix(),raw_source_exists=source.exists())
        if source.exists():
            A=np.loadtxt(source,skiprows=1);row.update(raw_source_sha256=sha(source),raw_samples=len(A),merged_samples=len(record['time']),same_samples=len(A)==len(record['time']),raw_channel_count=A.shape[1])
            if row['same_samples']and A.shape[1]==7:
                row['all_channels_match_round4']=True
                for j,c in enumerate(ORDER):
                    B=np.asarray(record[c]);row[c+'_maximum_absolute_difference']=float(np.max(np.abs(A[:,j]-B)));row['all_channels_match_round4']&=np.array_equal(np.round(A[:,j],4),B)
                row['timestamps_exact']=np.array_equal(A[:,2],np.asarray(record['time']))
                row['pen_states_exact']=np.array_equal(A[:,3],np.asarray(record['pen_status']))
        rows.append(row)
    return rows
def main():
    ids,y,schema,_,_=cohort();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3)as pool:
        for i,items in enumerate(pool.map(job,[(pid,schema['tasks'])for pid in ids])):
            rows.extend(items)
            if i%30==0:print('SVC SOURCES CHECKED',i+1,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'audit/raw_svc_comparison.csv',index=False)
    write(OUT/'audit/raw_svc_comparison.json',dict(complete=True,recordings=len(df),original_svc_files_available=int(df.raw_source_exists.sum()),
       identical_sample_counts=int(df.same_samples.fillna(False).sum()),identical_timestamps=int(df.timestamps_exact.fillna(False).sum()),identical_pen_states=int(df.pen_states_exact.fillna(False).sum()),
       merged_equals_original_rounded4=int(df.all_channels_match_round4.fillna(False).sum()),max_channel_difference={c:float(df[c+'_maximum_absolute_difference'].max())for c in ORDER},
       primary_input='Raw per-sample sequences in validated clean_data merged JSON (publication export precision); no feature aggregation or temporal resampling. Original SVC export preserved; this audit quantifies any precision difference.'))
    print('SVC PROVENANCE AUDIT COMPLETE',flush=True)

if __name__=='__main__':main()
