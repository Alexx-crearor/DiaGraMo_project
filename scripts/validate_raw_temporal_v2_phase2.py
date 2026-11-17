"""Held-out development replay, controlled perturbations, and frozen selection."""
import subprocess
from raw_temporal_v2_common import *
from raw_temporal_v2_training import SOURCES,predict,build_model
from train_raw_temporal_v2_phase2 import CONFIG_PATH,EXTRA
from diagramo.temporal_v2.multiscale import MultiScaleModel


def mask_windows(mode):
    rng=np.random.default_rng(42)
    def occlusion(batch):
        present=torch.ones(len(batch['length']),device=DEVICE,dtype=torch.bool)
        for person in range(batch['person_count']):
            rows=torch.nonzero(batch['patch_task']//14==person).flatten()
            valid=batch['sample_mask'][rows]& (batch['x'][rows,13,:]>.5)
            eligible=valid.any(1);indices=rows[eligible]
            if not len(indices):continue
            count=max(1,int(np.ceil(.05*len(indices))))
            if mode=='high_speed':
                score=(batch['x'][rows,10,:]*valid).sum(1)/valid.sum(1).clamp_min(1)
                chosen=indices[torch.argsort(score[eligible],descending=True,stable=True)[:count]]
            else:chosen=indices[torch.as_tensor(rng.choice(len(indices),count,replace=False),device=DEVICE)]
            present[chosen]=False
        return dict(patch_present=present)
    return occlusion


def main():
    seed_all(42);config=json.loads((ROOT/CONFIG_PATH).read_text());ids,y,schema,train,val,test=development()
    table=pd.read_csv(OUT/'tables/phase2_development_comparison.csv')
    primary=table[table.configuration.isin(['statistical','mil','stroke_transformer'])]
    grouped=primary.groupby('configuration')[['f1_macro','auc','parameters']].mean().reset_index()
    winner=grouped.sort_values(['f1_macro','auc','parameters'],ascending=[False,False,True]).iloc[0]
    reference=table[(table.configuration=='original_encoder_mil')&(table.seed==42)].iloc[0]
    multiscale=table[(table.configuration=='mil')&(table.seed==42)].iloc[0]
    local='original' if reference.f1_macro>multiscale.f1_macro and reference.auc>multiscale.auc else 'multiscale'
    decision=dict(selected_design=winner.configuration,rule=config['family_selection'],
        selected_local_encoder=local,local_encoder_rule=config['ssl_local_encoder_selection'],
        metrics=winner.to_dict(),outer_test_used=False,phase1_dependency_sha256=config['phase1_validation_sha256'])
    p1info=config['phase1_reference'];assert sha(ROOT/p1info['path'])==p1info['sha256']
    p1=json.loads((ROOT/p1info['path']).read_text());p1second=table[table.configuration=='p1_reference'].iloc[0]
    p1mean=dict(f1_macro=(p1['metrics']['f1_macro']+p1second.f1_macro)/2,
        auc=(p1['metrics']['auc']+p1second.auc)/2,parameters=p1['parameters'])
    p1wins=bool((p1mean['f1_macro'],p1mean['auc'],-p1mean['parameters'])>(winner.f1_macro,winner.auc,-winner.parameters))
    selected_settings=dict(config['training'])
    if p1wins:
        selected=dict(model_kind='recovery',**p1['architecture']);selected_local='original'
        selected_settings.update(next(c for c in config['candidates'] if c['name']=='p1_reference').get('training_overrides',{}))
    else:
        selected=dict(design=winner.configuration,relative_time=True,relative_spatial=True,index_only=False,
            dropout=.1,d_model=32,local_encoder=local,local_chunk_windows=256);selected_local=local
    decision.update(selected_architecture_for_ssl=selected,selected_local_encoder_for_ssl=selected_local,
        selected_training_settings_for_ssl=selected_settings,
        ssl_architecture_rule=config['ssl_architecture_selection'],p1_reference_two_seed_mean=p1mean,
        phase1_reference_selected_for_ssl=p1wins)
    write(OUT/'phase2_temporal/development_decision.json',decision)
    sig=signature(CONFIG_PATH,SOURCES+EXTRA);replay=[];perturbations=[];paired=[]
    for candidate in config['candidates']:
        eid=f"p2_{candidate['name']}_seed{candidate['seed']}";dest=OUT/'phase2_temporal'/eid
        result=json.loads((dest/'result.json').read_text());assert result['complete'] and result['signature']==sig['signature']
        assert set(result['fit_ids'])==set(ids[train]) and set(result['evaluation_ids'])==set(ids[val])
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
        saved=torch.load(ROOT/result['checkpoint'],map_location=DEVICE,weights_only=False)
        assert set(saved['normalizer'].fit_ids)==set(ids[train]);assert not set(saved['fit_ids'])&set(ids[test])
        model=build_model(saved['architecture']).to(DEVICE);model.load_state_dict(saved['state'])
        logits,z=predict(model,saved['normalizer'],ids[val]);p=1/(1+np.exp(-logits))
        old=pd.read_csv(dest/'evaluation_predictions.csv').set_index('participant_id').loc[ids[val]]
        error=float(np.max(np.abs(old.probability.values-p)));assert error<1e-6
        for key,value in metrics(y[val],p).items():assert abs(value-result['metrics'][key])<1e-8
        replay.append(dict(experiment=eid,replay_max_error=error,normalizer_train_only=True))
        curve=pd.read_csv(dest/'learning_curve.csv');assert np.isfinite(curve.gradient_norm).all() and (curve.local_gradient_norm>0).all()
        for stress in (() if saved['architecture'].get('model_kind')=='recovery' else ('transport','content_only','high_speed','random_windows')):
            changed,_=predict(model,saved['normalizer'],ids[val],stress=stress if stress in ('transport','content_only') else None,
                occlusion=mask_windows(stress) if stress in ('high_speed','random_windows') else None)
            probability=1/(1+np.exp(-changed));delta=probability-p
            if stress=='transport':assert np.max(np.abs(delta))<1e-5,'Metadata-consistent transport should be invariant'
            perturbations.append(dict(configuration=candidate['name'],seed=candidate['seed'],stress=stress,
                mean_abs_probability_change=float(np.abs(delta).mean()),**metrics(y[val],probability)))
            paired.extend(dict(participant_id=pid,label=int(label),configuration=candidate['name'],seed=candidate['seed'],
                stress=stress,baseline_probability=float(a),perturbed_probability=float(b),delta=float(b-a))
                for pid,label,a,b in zip(ids[val],y[val],p,probability))
        print('PHASE2 VALIDATED',eid,flush=True)
    pd.DataFrame(replay).to_csv(OUT/'tables/phase2_checkpoint_replay.csv',index=False)
    pd.DataFrame(perturbations).to_csv(OUT/'tables/phase2_perturbations.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'phase2_temporal/paired_perturbation_predictions.csv',index=False)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_raw_temporal_v2_multiscale.py','-q'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'phase2_temporal/unit_tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8');assert tests.returncode==0
    write(OUT/'phase2_validation.json',dict(complete=True,development_fits=len(replay),checkpoint_replay_passed=True,
        transport_equivariance_passed=True,unit_tests_passed=True,outer_test_loaded=False,
        selected_before_perturbation_scoring=True,selection=decision,phase3_ready=True,
        interpretation='Exploratory development only. Content-only reversal is inconsistent-metadata OOD stress, not a causal manipulation.'))


if __name__=='__main__':
    with gpu_worker():main()
