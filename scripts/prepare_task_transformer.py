"""Build label-free raw perturbation banks without touching baseline caches."""
import json
import time
import hashlib
import concurrent.futures
import numpy as np
from task_transformer_common import ROOT, OUT, CONFIG, initialize, data, sha, write
from prepare import canonical
from diagramo.branches.kinematic import extract
from diagramo.motor_nuisance import motor_normalize, perturb_record


def participant_job(item):
    pid, tasks, feature_names, original_X, names = item
    partial = OUT / 'cache/participants' / (pid + '.npz')
    source = ROOT / 'data/clean_data' / (pid + '_merged.json')
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if partial.exists():
        d = np.load(partial)
        assert str(d['source_sha256']) == digest
        return pid, {name:d[name] for name in names}, json.loads(str(d['audits'])), digest
    raw = json.loads(payload)
    by_task = {canonical(k): v['data'] for k,v in raw.items() if canonical(k) in tasks}
    bank = {name:np.zeros(1036,dtype='float32') for name in names}
    audits=[]
    for ti,task in enumerate(tasks):
        record=by_task[task]
        original,actual_names=extract(record)
        assert actual_names==feature_names
        np.testing.assert_array_equal(original,original_X[ti*74:(ti+1)*74])
        bank['motor_clean'][ti*74:(ti+1)*74]=extract(motor_normalize(record))[0]
        for nuisance in CONFIG['nuisances']:
            for training in (True,False):
                transformed=perturb_record(record,nuisance,training=training)
                bank[('train_' if training else 'stress_')+nuisance][ti*74:(ti+1)*74]=extract(transformed)[0]
                if not training:
                    bank['motor_stress_'+nuisance][ti*74:(ti+1)*74]=extract(motor_normalize(transformed))[0]
                    t=np.asarray(record['time']);u=np.asarray(transformed['time'])
                    if nuisance!='sampling_half':
                        assert np.array_equal(np.sign(np.diff(t)),np.sign(np.diff(u)))
                    audits.append(dict(participant_id=pid,task=task,nuisance=nuisance,n_original=len(t),
                        n_transformed=len(u),duration_original=float(t[-1]-t[0]),duration_transformed=float(u[-1]-u[0]),
                        finite=True,label_preservation_proven=False))
    partial.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(partial,source_sha256=digest,audits=json.dumps(audits),**bank)
    return pid,bank,audits,digest


def main():
    initialize()
    target = OUT / 'cache/features.npz'
    signature = {p: sha(ROOT / p) for p in ('src/diagramo/motor_nuisance.py', 'scripts/prepare_task_transformer.py', 'src/diagramo/features.py')}
    if target.exists():
        assert json.loads((OUT / 'cache/code.json').read_text()) == signature
        print('REUSE raw nuisance feature bank')
        return
    X, y, ids, schema, assignments, oof, splits = data()
    tasks = schema['tasks']
    names = ['motor_clean'] + [p + n for p in ('train_', 'stress_', 'motor_stress_') for n in CONFIG['nuisances']]
    bank = {name: np.zeros_like(X) for name in names}
    audits, sources = [], []
    began = time.perf_counter()
    jobs=[(str(pid),tasks,schema['features_per_task'],X[i],names) for i,pid in enumerate(ids)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for i,(pid,partial,checks,digest) in enumerate(pool.map(participant_job,jobs)):
            assert pid==ids[i]
            for name in names:bank[name][i]=partial[name]
            audits.extend(checks)
            sources.append(dict(participant_id=pid,path='data/clean_data/'+pid+'_merged.json',sha256=digest))
            if i%10==0:print(f'RAW CACHE {i+1}/258; {time.perf_counter()-began:.1f}s',flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, raw_clean=X, ids=ids, y=y, **bank)
    import pandas as pd
    pd.DataFrame(audits).to_csv(OUT / 'cache/raw_transform_checks.csv', index=False)
    assignments.to_csv(OUT / 'fold_assignments.csv', index=False)
    write(OUT / 'cache/source_hashes.json', sources)
    write(OUT / 'cache/code.json', signature)
    write(OUT / 'cache/validation.json', dict(complete=True, original_cache_exactly_reconstructed=True,
         subjects=258, tasks=14, transformed_banks=len(bank), checked_raw_record_conditions=len(audits),
         seconds=time.perf_counter()-began, feature_bank_sha256=sha(target),
         augmentation_use='training banks may only be indexed with current fit IDs; stress banks held out for diagnosis'))
    print('RAW CACHE COMPLETE', time.perf_counter()-began, flush=True)


if __name__ == '__main__':
    main()
