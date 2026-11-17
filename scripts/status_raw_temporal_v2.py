"""Read live artifact status without fitting or inferring phase completion."""
from raw_temporal_v2_common import *


if __name__=='__main__':
    status=json.loads((OUT/'validation.json').read_text());print(json.dumps(status,ensure_ascii=False,indent=2))
    paths=[('Phase0',OUT/'phase0_diagnostics','p0_*/result.json'),
           ('Phase1',OUT/'phase1_bottleneck','p1_*/result.json'),
           ('Phase2',OUT/'phase2_temporal','p2_*/result.json'),
           ('Phase3',OUT/'phase3_ssl','*/result.json'),
           ('Phase4',OUT/'phase4_complementarity','*/result.json')]
    rows=[]
    for phase,folder,pattern in paths:
        records=[json.loads(p.read_text()) for p in folder.glob(pattern)]
        print(phase,'completed result files',sum(r.get('complete',False) for r in records))
        rows.extend(dict(phase=phase,experiment=r['experiment_id'],**r['metrics']) for r in records if 'metrics' in r)
    if rows:print(pd.DataFrame(rows).to_string(index=False))
    lock=LOCAL/'gpu_worker.lock'
    print('GPU lock:',lock.read_text() if lock.exists() else 'none')
