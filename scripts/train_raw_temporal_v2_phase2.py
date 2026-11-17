"""Execute only the Phase2 candidate list frozen after Phase1."""
from raw_temporal_v2_common import *
from raw_temporal_v2_training import fit,build_model,tensor
from diagramo.temporal.dataset import TrainNormalizer,collate_people

CONFIG_PATH='configs/raw_temporal_v2/phase2.json'
EXTRA=['scripts/train_raw_temporal_v2_phase2.py','scripts/configure_raw_temporal_v2_phase2.py',
       'scripts/train_raw_temporal_v2_phase1.py',
       'results/raw_temporal_v2/phase1_validation.json','results/raw_temporal_v2/phase1_bottleneck/development_decision.json']


def main():
    config=json.loads((ROOT/CONFIG_PATH).read_text());assert json.loads((OUT/'phase1_validation.json').read_text())['complete']
    assert config['phase1_validation_sha256']==sha(OUT/'phase1_validation.json')
    assert json.loads((OUT/'phase2_temporal/relationship_manifest.json').read_text())['complete']
    progress=json.loads((OUT/'validation.json').read_text());progress.update(phase2='in_progress',complete=False);write(OUT/'validation.json',progress)
    ids,y,schema,train,val,test=development();rows=[]
    # Engineering only: worst two development-Train bags, no labels/objective
    # fitting. Check the real-data ragged batch on the actual single GPU.
    seed_all(42)
    physical=json.loads((OUT/'phase1_bottleneck/physical_targets_manifest.json').read_text())
    sizes={r['participant_id']:r['windows'] for r in physical['records']}
    chosen=sorted(ids[train],key=lambda pid:sizes[pid],reverse=True)[:2]
    with np.load(CACHE/'statistics.npz') as source:statistics=source['statistics']
    normalizer=TrainNormalizer().fit(statistics,train,ids[train]);items=[tensor(pid,normalizer) for pid in chosen]
    engineering=[]
    for design in ('statistical','mil','stroke_transformer'):
        architecture=dict(design=design,local_chunk_windows=256);model=build_model(architecture).to(DEVICE).train()
        if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
        output=model(collate_people(items));output.sum().backward()
        assert torch.isfinite(output).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(float(p.grad.abs().sum()) for p in model.local.parameters() if p.grad is not None)>0
        model.eval()
        with torch.no_grad():
            together=model(collate_people(items));separate=torch.cat([model(collate_people([item])) for item in items])
        error=float((together-separate).abs().max());assert error<1e-4
        resource_check();engineering.append(dict(design=design,train_ids=chosen,batch_max_logit_difference=error,
            peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2 if DEVICE=='cuda' else 0.,gradient_finite=True,no_optimizer_step=True))
        del model
    del items
    write(OUT/'phase2_temporal/real_model_engineering.json',dict(complete=True,scope='Train-only forward/backward engineering, no model fit',records=engineering))
    for candidate in config['candidates']:
        if candidate['architecture'].get('model_kind')=='recovery':architecture=dict(candidate['architecture'])
        else:
            architecture=dict(design='mil',relative_time=True,relative_spatial=True,index_only=False,dropout=.1,d_model=32,local_encoder='multiscale',local_chunk_windows=256)
            architecture.update(candidate['architecture'])
        settings=dict(config['training']);settings.update(candidate.get('training_overrides',{}))
        result=fit(f"p2_{candidate['name']}_seed{candidate['seed']}",'2',CONFIG_PATH,settings,
            architecture,train,validation=val,seed=candidate['seed'],extra_sources=EXTRA)
        rows.append(dict(configuration=candidate['name'],seed=candidate['seed'],**result['metrics'],
            train_f1_macro=result['train_metrics']['f1_macro'],parameters=result['parameters'],
            best_epoch=result['best_epoch'],epochs=result['epochs_run'],seconds=result['training_seconds'],peak_gpu_mib=result['peak_gpu_mib']))
        pd.DataFrame(rows).to_csv(OUT/'tables/phase2_development_comparison.csv',index=False)


if __name__=='__main__':
    with gpu_worker():main()
