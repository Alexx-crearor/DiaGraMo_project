"""Two-seed masked SSL, matched scratch/fine-tuning, frozen readouts, label curves."""
from raw_temporal_v2_common import *
from raw_temporal_v2_ssl import pretrain
from raw_temporal_v2_training import fit
from raw_temporal_v2_frozen import features,fit_readout
from configure_raw_temporal_v2_phase3 import labeled_subset

CONFIG_PATH='configs/raw_temporal_v2/phase3.json'
EXTRA=['scripts/train_raw_temporal_v2_phase3.py','scripts/configure_raw_temporal_v2_phase3.py',
    'results/raw_temporal_v2/phase2_validation.json','results/raw_temporal_v2/phase2_temporal/development_decision.json']


def main():
    config=json.loads((ROOT/CONFIG_PATH).read_text());assert config['source_phase2_validation_sha256']==sha(OUT/'phase2_validation.json')
    ids,y,schema,train,val,test=development();rows=[];pretrained={};feature_paths={}
    progress=json.loads((OUT/'validation.json').read_text());progress.update(phase3='in_progress',complete=False);write(OUT/'validation.json',progress)
    def append(result,method,seed,hyper=None,fraction=1.,scope='full_supervision'):
        rows.append(dict(experiment=result['experiment_id'],method=method,seed=seed,hyper=hyper,
            labeled_fraction=fraction,labeled_train_participants=len(result['fit_ids']),validation_participants=len(val),
            trainable_parameters=result['parameters'],
            scope=scope,**result['metrics'],seconds=result['training_seconds']))
        pd.DataFrame(rows).to_csv(OUT/'tables/phase3_all_downstream_fits.csv',index=False)
    for seed in config['seeds']:
        ssl=pretrain(f'p3_masked_ssl_seed{seed}',CONFIG_PATH,config['pretraining'],train,validation=val,seed=seed,extra_sources=EXTRA)
        checkpoint=ROOT/ssl['checkpoint'];pretrained[seed]=checkpoint
        for method in ('scratch','finetune'):
            result=fit(f'p3_{method}_seed{seed}','3',CONFIG_PATH,config['supervised_training'],config['architecture'],
                train,validation=val,seed=seed,extra_sources=EXTRA,pretrained_local=checkpoint if method=='finetune' else None)
            append(result,method,seed)
        for random in (False,True):
            tag='random' if random else 'ssl'
            feature_path=features(f'p3_{tag}_seed{seed}',checkpoint,train,val,random=random,seed=seed)
            feature_paths[(seed,tag)]=feature_path
            for kind,values in config['frozen_readout_grid'].items():
                for hyper in values:
                    result=fit_readout(f'p3_frozen_{tag}_{kind}_hyper{hyper}_seed{seed}',CONFIG_PATH,feature_path,
                        train,val,kind,hyper,seed=seed,extra_sources=EXTRA)
                    append(result,'frozen_'+tag+'_'+kind,seed,hyper)
    sensitivity=config['label_sensitivity']
    for seed in sensitivity['neural_seeds']:
        for fraction in sensitivity['fractions']:
            if fraction==1:continue
            chosen=labeled_subset(train,y,fraction,seed)
            assert ids[chosen].tolist()==config['labeled_subsets'][f'seed{seed}_fraction{fraction}']
            for method in ('scratch','finetune'):
                result=fit(f'p3_label_{method}_fraction{fraction}_seed{seed}','3',CONFIG_PATH,
                    config['supervised_training'],config['architecture'],chosen,validation=val,seed=seed,
                    normalization_train=train,extra_sources=EXTRA,pretrained_local=pretrained[seed] if method=='finetune' else None)
                append(result,method,seed,fraction=fraction,scope='label_sensitivity')
    for seed in sensitivity['frozen_seeds']:
        for fraction in sensitivity['fractions']:
            chosen=labeled_subset(train,y,fraction,seed)
            for tag in ('ssl','random'):
                result=fit_readout(f'p3_label_frozen_{tag}_fraction{fraction}_seed{seed}',CONFIG_PATH,
                    feature_paths[(sensitivity['frozen_encoder_seed'],tag)],chosen,val,
                    sensitivity['frozen_kind'],sensitivity['frozen_hyper'],seed=seed,extra_sources=EXTRA)
                append(result,'frozen_'+tag+'_linear',seed,sensitivity['frozen_hyper'],fraction,scope='label_sensitivity')
    table=pd.DataFrame(rows);table[table.scope=='label_sensitivity'].to_csv(OUT/'tables/phase3_label_sensitivity.csv',index=False)
    full=table[table.scope=='full_supervision'];selected=[]
    for method,group in full.groupby('method'):
        if method in ('scratch','finetune'):selected.extend(group.to_dict('records'));continue
        settings=group.groupby('hyper')[['f1_macro','auc']].mean().reset_index().sort_values(['f1_macro','auc','hyper'],ascending=[False,False,True])
        # Freeze SSL head settings and apply the same settings to its random control.
        if method.startswith('frozen_random_'):continue
        best=settings.iloc[0].hyper
        selected.extend(group[group.hyper==best].to_dict('records'))
        counterpart=method.replace('frozen_ssl_','frozen_random_');selected.extend(full[(full.method==counterpart)&(full.hyper==best)].to_dict('records'))
    pd.DataFrame(selected).to_csv(OUT/'tables/phase3_ssl_comparison.csv',index=False)
    print('PHASE3 FITS FINISHED; VALIDATION PENDING',len(rows),flush=True)


if __name__=='__main__':
    with gpu_worker():main()
