"""Six frozen outer folds, Train-only inner selection, two final initializations."""
from raw_temporal_v2_common import *
from raw_temporal_v2_evaluation import config,neural,frozen_ssl,kinematic,probability,SOURCES,CONFIG_PATH


def run_outer(fold):
    cfg=config();ids,y,schema,assignments,splits=cohort();train,test=splits[fold]
    destination=OUT/'phase4_complementarity'/f'outer_{fold}';destination.mkdir(parents=True,exist_ok=True)
    inner=list(StratifiedKFold(3,shuffle=True,random_state=42).split(train,y[train]));search=[];final=[]
    branches=['raw_v2','ssl_v2']+(['random_frozen_v2'] if cfg['preferred_ssl_method'].startswith('frozen_ssl_') else [])
    for branch in branches:
        method=cfg['preferred_ssl_method']
        frozen=branch in ('ssl_v2','random_frozen_v2') and method.startswith('frozen_ssl_')
        kind=method.removeprefix('frozen_ssl_') if frozen else None
        candidates=cfg['frozen_readout_grid'][kind] if frozen else cfg['raw_learning_rates']
        candidate_rows=[]
        for hyper in candidates:
            rows=[]
            for j,(a,b) in enumerate(inner):
                fit=train[a];validation=train[b]
                if frozen:result=frozen_ssl(fit,validation,kind,hyper,42,random=branch=='random_frozen_v2')
                else:result=neural(f'p4_outer{fold}_{branch}_inner{j}_lr{hyper}',fit,validation,hyper,42,ssl=branch=='ssl_v2')
                row=dict(outer_fold=fold,branch=branch,hyper=hyper,inner_fold=j,**result['metrics'],
                    best_epoch=result.get('best_epoch'),experiment=result['experiment_id'],checkpoint=result['checkpoint'])
                rows.append(row);search.append(row)
                pd.DataFrame(search).to_csv(destination/'inner_search.csv',index=False)
            candidate_rows.append(dict(hyper=hyper,macro_f1=float(np.mean([r['f1_macro'] for r in rows])),
                auc=float(np.mean([r['auc'] for r in rows])),epochs=[r['best_epoch'] for r in rows] if not frozen else []))
        best=max(candidate_rows,key=lambda r:(r['macro_f1'],r['auc'],r['hyper'] if frozen and kind=='nonlinear' else -r['hyper']))
        epochs=None if frozen else max(1,int(round(np.median(best['epochs']))))
        selection=dict(outer_fold=fold,branch=branch,hyper=best['hyper'],final_epochs=epochs,
            candidates=candidate_rows,fit_ids=ids[train].tolist(),test_ids=ids[test].tolist(),
            selection_used_only_outer_train=True)
        write(destination/(branch+'_selection.json'),selection)
        for seed in cfg['initialization_seeds']:
            if frozen:result=frozen_ssl(train,test,kind,best['hyper'],seed,random=branch=='random_frozen_v2')
            else:result=neural(f'p4_outer{fold}_{branch}_final_seed{seed}',train,test,best['hyper'],seed,
                fixed_epochs=epochs,ssl=branch=='ssl_v2')
            p=probability(result,test)
            final.extend(dict(participant_id=pid,label=int(label),fold=fold,model=branch,seed=seed,probability=float(value),
                prediction=int(value>.5),checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],
                experiment_id=result['experiment_id']) for pid,label,value in zip(ids[test],y[test],p))
    kin=kinematic(train,test)
    pd.DataFrame(final).to_csv(destination/'nested_oof.csv',index=False)
    write(destination/'nested_fit_status.json',dict(fits_complete=True,full_validation_pending=True,outer_fold=fold,participants=len(test),
        models=branches,
        raw_and_ssl_final_seeds=cfg['initialization_seeds'],selection_outer_train_only=True,
        baseline_checkpoint_reused=kin['reused_original_checkpoint']))
    print('OUTER NESTED COMPLETE',fold,flush=True)


def main():
    cfg=config();assert cfg['fold_assignments_sha256']==sha(ROOT/'results/cv/fold_assignments.csv')
    for path,digest in cfg['dependencies'].items():assert sha(ROOT/path)==digest
    progress=json.loads((OUT/'validation.json').read_text());progress.update(phase4='in_progress',complete=False);write(OUT/'validation.json',progress)
    for fold in range(6):run_outer(fold)
    frame=pd.concat([pd.read_csv(OUT/'phase4_complementarity'/f'outer_{fold}/nested_oof.csv') for fold in range(6)],ignore_index=True)
    expected_models=3 if cfg['preferred_ssl_method'].startswith('frozen_ssl_') else 2
    assert len(frame)==258*2*expected_models and not frame.duplicated(['participant_id','model','seed']).any()
    frame.to_csv(OUT/'phase4_complementarity/nested_oof.csv',index=False)
    print('SIX-FOLD RAW/SSL NESTED FITS FINISHED; FUSION/XAI/VALIDATION PENDING',flush=True)


if __name__=='__main__':
    with gpu_worker():main()
