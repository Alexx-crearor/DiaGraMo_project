"""Replay nested choices, train-only transforms, OOF, robustness and logit IG."""
import argparse
import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from task_transformer_common import ROOT,OUT,CONFIG,FAMILIES,METRICS,data,cache,device,features_for,sha,write,score,training_signature
from train_task_transformer import load_model,predict,task_mask
from diagramo.task_transformer import TaskPreprocessor


def close(a,b,atol=1e-12):np.testing.assert_allclose(a,b,atol=atol,rtol=0)


def protect():
    manifest=json.loads((OUT/'protected_manifest.json').read_text())
    for rel,digest in manifest.items():assert sha(ROOT/rel)==digest,rel
    assert json.loads((ROOT/'results/kinematic_xai/validation.json').read_text())['complete']
    training_signature()
    return len(manifest)


def check_pre(pre,X):
    independent=TaskPreprocessor().fit(X)
    assert pre.n_fit==len(X)
    np.testing.assert_array_equal(pre.imputer.statistics_,independent.imputer.statistics_)
    np.testing.assert_array_equal(pre.scaler.mean_,independent.scaler.mean_)
    np.testing.assert_array_equal(pre.scaler.scale_,independent.scaler.scale_)


def numerics():
    protected=protect()
    X,y,ids,schema,assignment,original,splits=data();bank=cache();dev=device()
    cache_validation=json.loads((OUT/'cache/validation.json').read_text())
    assert cache_validation['complete'] and sha(OUT/'cache/features.npz')==cache_validation['feature_bank_sha256']
    for rel,digest in json.loads((OUT/'cache/code.json').read_text()).items():assert sha(ROOT/rel)==digest,rel
    np.testing.assert_array_equal(bank['raw_clean'],X);np.testing.assert_array_equal(bank['ids'],ids)
    for key in bank.files:
        if key not in ('ids','y'):assert bank[key].shape==(258,1036) and np.isfinite(bank[key]).all()
    for row in json.loads((OUT/'cache/source_hashes.json').read_text()):assert sha(ROOT/row['path'])==row['sha256']
    inner_count,outer_count,ig_count,permutation_count=0,0,0,0
    code_hashes={p.relative_to(ROOT).as_posix():sha(p) for folder in ('scripts','src/diagramo') for p in (ROOT/folder).glob('*task_transformer*.py')}
    code_hashes['src/diagramo/motor_nuisance.py']=sha(ROOT/'src/diagramo/motor_nuisance.py')
    for rel in ('configs/task_transformer.json','configs/task_transformer_controls.json','tests/test_task_transformer.py'):
        code_hashes[rel]=sha(ROOT/rel)
    for f,(tr,te) in enumerate(splits):
        folder=OUT/'folds'/f'fold_{f}'
        assert json.loads((folder/'complete.json').read_text())['complete']
        expected=list(StratifiedKFold(3,shuffle=True,random_state=42).split(X[tr],y[tr]))
        search=[]
        for j,(fit,val) in enumerate(expected):
            p=OUT/'train_only'/f'outer_{f}'/f'inner_{j}'/'provenance.json'
            rankinfo=json.loads(p.read_text())
            assert rankinfo['fit_ids']==ids[tr[fit]].tolist()
            assert set(rankinfo['forbidden_ids'])==set(ids[np.r_[tr[val],te]])
            assert not set(rankinfo['fit_ids'])&set(rankinfo['forbidden_ids'])
            seen=[]
            for item in rankinfo['inner_splits']:
                assert set(item['train_ids'])<=set(rankinfo['fit_ids'])
                assert set(item['validation_ids'])<=set(rankinfo['fit_ids'])
                assert not set(item['train_ids'])&set(item['validation_ids'])
                seen+=item['validation_ids']
            assert sorted(seen)==sorted(rankinfo['fit_ids'])
        for family in FAMILIES:
            feature=features_for(family,bank)
            for ci,candidate in enumerate(CONFIG['candidates']):
                for j,(fit,val) in enumerate(expected):
                    info=json.loads((folder/'inner'/f'{family}_{ci}_{j}.json').read_text())
                    assert info['train_ids']==ids[tr[fit]].tolist() and info['validation_ids']==ids[tr[val]].tolist()
                    assert not set(info['train_ids'])&set(ids[te])
                    assert not set(info['validation_ids'])&set(ids[te])
                    assert sha(ROOT/info['checkpoint'])==info['checkpoint_sha256']
                    model,b=load_model(ROOT/info['checkpoint'],dev)
                    assert b['train_ids'].tolist()==info['train_ids'] and b['validation_ids'].tolist()==info['validation_ids']
                    check_pre(b['preprocessor'],feature[tr[fit]])
                    ranking=pd.read_csv(OUT/'train_only'/f'outer_{f}'/f'inner_{j}'/'task_importance.csv')
                    np.testing.assert_array_equal(b['present'],task_mask(family,candidate,schema,ranking))
                    prob,_=predict(model,b['preprocessor'],feature[tr[val]],b['present'],dev)
                    saved=pd.read_csv(folder/'inner'/f'{family}_{ci}_{j}.oof.csv',float_precision='round_trip')
                    np.testing.assert_array_equal(saved.participant_id,ids[tr[val]])
                    close(saved.probability,prob,atol=2e-6)
                    np.testing.assert_array_equal(saved.prediction,(prob>.5).astype(int))
                    for k,value in score(saved.label,saved.probability,saved.prediction).items():close(info[k],value)
                    history=json.loads((folder/'inner'/f'{family}_{ci}_{j}_history.json').read_text())
                    assert 1<=info['best_epoch']<=len(history)<=CONFIG['training']['max_epochs']
                    chosen=next(row for row in history if row['epoch']==info['best_epoch'])
                    assert chosen['validation_loss']<=min(row['validation_loss'] for row in history)+1e-5
                    search.append(info);inner_count+=1
        average=pd.DataFrame(search).groupby(['family','candidate'],sort=False).agg(
            f1_macro=('f1_macro','mean'),auc=('auc','mean'),parameters=('parameters','first'),epochs=('best_epoch','median')).reset_index()
        selection=json.loads((folder/'selection.json').read_text())
        for family in FAMILIES:
            best=average[average.family==family].sort_values(['f1_macro','auc','parameters','candidate'],ascending=[False,False,True,True]).iloc[0]
            assert selection['families'][family]['candidate']==best.candidate
            assert selection['families'][family]['epochs']==max(1,int(best.epochs))
        best=average.sort_values(['f1_macro','auc','parameters','candidate','family'],ascending=[False,False,True,True,True]).iloc[0]
        assert selection['nested_selected_family']==best.family and selection['nested_selected_candidate']==best.candidate
        outerrank=pd.read_csv(OUT/'train_only'/f'outer_{f}'/'final_train/task_importance.csv')
        rankinfo=json.loads((OUT/'train_only'/f'outer_{f}'/'final_train/provenance.json').read_text())
        assert rankinfo['fit_ids']==ids[tr].tolist() and rankinfo['forbidden_ids']==ids[te].tolist()
        for parent in (folder/'outer',OUT/'controls'/f'fold_{f}'):
            for p in parent.glob('*.oof.csv'):
                d=pd.read_csv(p,float_precision='round_trip')
                info=json.loads(p.with_name(p.name.replace('.oof.csv','.json')).read_text())
                assert info['train_ids']==ids[tr].tolist() and info['test_ids']==ids[te].tolist()
                np.testing.assert_array_equal(d.participant_id,ids[te]);np.testing.assert_array_equal(d.label,y[te])
                assert sha(ROOT/info['checkpoint'])==info['checkpoint_sha256']
                if info['model'].startswith('lightgbm'):
                    model=joblib.load(ROOT/info['checkpoint']);columns=np.array(info['columns'])
                    close(model[0].statistics_,np.median(X[tr][:,columns],axis=0),atol=1e-5)
                    prob=model.predict_proba(X[te][:,columns])[:,1]
                    if info['model']=='lightgbm' and info['seed']==42:close(prob,original.probability.to_numpy()[te])
                else:
                    family=info['family'];candidate=CONFIG['candidates'][info['candidate']]
                    model,b=load_model(ROOT/info['checkpoint'],dev)
                    assert b['train_ids'].tolist()==ids[tr].tolist() and b['validation_ids'].tolist()==ids[te].tolist()
                    feature=features_for(family,bank);check_pre(b['preprocessor'],feature[tr])
                    np.testing.assert_array_equal(b['present'],task_mask(family,candidate,schema,outerrank))
                    expected_epochs=max(1,int(average[(average.family==family)&(average.candidate==info['candidate'])].iloc[0].epochs))
                    assert info['epochs']==info['epochs_ran']==expected_epochs
                    prob,_=predict(model,b['preprocessor'],feature[te],b['present'],dev)
                    assert sum(v.numel() for v in model.parameters())==info['parameters']<100000
                close(prob,d.probability,atol=2e-6);np.testing.assert_array_equal(d.prediction,(prob>.5).astype(int))
                for k,value in score(d.label,d.probability,d.prediction).items():close(info[k],value)
                outer_count+=1
        analysis=OUT/'analysis'/f'fold_{f}'
        assert json.loads((analysis/'complete.json').read_text())['complete']
        robust=pd.read_csv(analysis/'robustness_oof.csv',float_precision='round_trip')
        masks=pd.read_csv(analysis/'masking_oof.csv',float_precision='round_trip')
        zeros=pd.read_csv(analysis/'feature_ablation_oof.csv',float_precision='round_trip')
        permutation=pd.read_csv(analysis/'permutation.csv',float_precision='round_trip')
        attention_table=pd.read_csv(analysis/'attention.csv',float_precision='round_trip')
        for label in ['lightgbm','lightgbm_without_text',*FAMILIES]:
            info=json.loads((folder/'outer'/f'{label}_seed_42.json').read_text())
            if label.startswith('lightgbm'):
                model=joblib.load(ROOT/info['checkpoint']);columns=np.array(info['columns']);feature=X
                present=np.zeros(14,bool);present[np.unique(columns//74)]=True
                def forward(A,mask=None):
                    test=np.array(A,copy=True)
                    if mask is not None:
                        local=np.flatnonzero(np.isin(columns,np.flatnonzero(np.repeat(~mask,74))))
                        test[:,columns[local]]=model[0].statistics_[local]
                    return model.predict_proba(test[:,columns])[:,1]
            else:
                feature=features_for(label,bank);model,b=load_model(ROOT/info['checkpoint'],dev);pre=b['preprocessor'];present=b['present']
                def forward(A,mask=None):return predict(model,pre,A,present if mask is None else mask,dev)[0]
                for suffix in ['']+(['_'+n for n in CONFIG['nuisances']]if label in ('transformer','motor_normalized')else[]):
                    ig=np.load(analysis/(label+'_ig'+suffix+'.npz'))
                    np.testing.assert_array_equal(ig['ids'],ids[te]);np.testing.assert_array_equal(ig['present'],present)
                    Z=pre.transform(feature[te]if not suffix else features_for(label,bank,suffix[1:])[te])
                    with torch.no_grad():
                        z=torch.tensor(Z,device=dev);m=torch.tensor(present,device=dev).expand(len(te),-1)
                        logits=model(z,m).cpu().numpy();reference=model(torch.zeros_like(z),m).cpu().numpy()
                    close(ig['logits'],logits,atol=1e-5);close(ig['base'],reference,atol=1e-5)
                    error=ig['phi'].sum(axis=1)-(ig['logits']-ig['base'])
                    close(error,ig['error'])
                    tolerance=CONFIG['integrated_gradients_absolute_tolerance']+CONFIG['integrated_gradients_relative_tolerance']*np.abs(ig['logits']-ig['base'])
                    assert np.all(np.abs(error)<=tolerance)
                    assert np.all(ig['phi'].reshape(-1,14,74)[:,~present]==0)
                    ig_count+=1
                _,extra=predict(model,pre,feature[te],present,dev,return_attention=True)
                if extra['attention'].size:
                    weights=extra['attention'].mean(axis=(0,2))
                    incoming=(weights*present[None,:,None]).sum(1)/present.sum()
                    a=attention_table[attention_table.model==label]
                    close(incoming.reshape(-1),a.incoming_attention,atol=2e-6)
                    close(extra['pool_weights'].reshape(-1),a.pool_weight,atol=2e-6)
                    close(incoming.sum(1),np.ones(len(te)),atol=2e-6)
                    assert not np.any(incoming[:,~present])
            if label=='lightgbm':
                from diagramo.kinematic_xai import explain_pipeline
                explanation=explain_pipeline(model,X[te])
                with np.load(analysis/'lightgbm_shap.npz')as stored:
                    close(stored['phi'],explanation['phi']);close(stored['base'],explanation['base']);close(stored['logits'],explanation['raw_score'])
            clean=forward(feature[te]);baseline=score(y[te],clean)
            for nuisance in CONFIG['nuisances']:
                A=bank['stress_'+nuisance][te]if label.startswith('lightgbm')else features_for(label,bank,nuisance)[te]
                prob=forward(A)
                d=robust[(robust.model==label)&(robust.condition==nuisance)]
                np.testing.assert_array_equal(d.participant_id,ids[te]);close(d.probability,prob,atol=2e-6)
            for (kind,name,repeat),d in masks[masks.model==label].groupby(['kind','removed','repeat']):
                if kind=='one_task':removed=[schema['tasks'].index(name)]
                elif kind=='without_text_diagnostic':removed=[schema['tasks'].index(t)for t in ('copy','dictation')]
                else:
                    k=int(kind.removeprefix('missing_'));seed=int(np.random.SeedSequence([42,f,733,k,repeat]).generate_state(1)[0])
                    removed=np.sort(np.random.default_rng(seed).choice(14,k,replace=False))
                    assert name=='|'.join(schema['tasks'][i]for i in removed)
                mask=present.copy();mask[removed]=False
                close(forward(feature[te],mask),d.probability,atol=2e-6)
            for ti,task in enumerate(schema['tasks']):
                d=zeros[(zeros.model==label)&(zeros.task==task)]
                if label.startswith('lightgbm'):
                    test=np.array(feature[te],copy=True);test[:,ti*74:(ti+1)*74]=X[tr,ti*74:(ti+1)*74].mean(axis=0,dtype=np.float64);prob=forward(test)
                else:
                    Z=pre.transform(feature[te]);Z[:,ti]=0.
                    with torch.no_grad():prob=torch.sigmoid(model(torch.tensor(Z,device=dev),torch.tensor(present,device=dev).expand(len(te),-1))).cpu().numpy()
                close(prob,d.probability,atol=2e-6)
            perm=permutation[permutation.model==label]
            from diagramo.kinematic_xai import feature_mapping,blocks
            mapping=feature_mapping(schema['tasks'],schema['features_per_task'])
            with np.load(analysis/(label+'_permutations.npz'))as saved:
                assert len(perm)==260
                for i,row in enumerate(perm.itertuples()):
                    li=('task','group').index(row.level);bi=list(blocks(mapping,row.level)).index(row.block)
                    seed=int(np.random.SeedSequence([42,f,991,li,bi,row.repeat]).generate_state(1)[0]);assert seed==row.seed
                    order=np.random.default_rng(seed).permutation(len(te));np.testing.assert_array_equal(order,saved['orders'][i])
                    A=np.array(feature[te],copy=True);columns_block=blocks(mapping,row.level)[row.block];A[:,columns_block]=feature[te][order][:,columns_block]
                    prob=forward(A);close(prob,saved['probability'][i],atol=2e-6)
                    for k,v in score(y[te],saved['probability'][i],saved['prediction'][i]).items():
                        close(v,getattr(row,k));close(baseline[k]-v,getattr(row,'drop_'+k))
                    permutation_count+=1
        print('VALIDATED outer',f,'nested choices, checkpoints, all nuisance/mask/permutation predictions and IG',flush=True)
    assert inner_count==432 and outer_count==180 and ig_count==108 and permutation_count==12480
    oof=pd.read_csv(OUT/'tables/oof_predictions.csv',float_precision='round_trip')
    summary=pd.read_csv(OUT/'tables/performance_summary.csv')
    for(model,seed),d in oof.groupby(['model','seed']):
        assert len(d)==258 and d.participant_id.is_unique
        row=summary[(summary.model==model)&(summary.seed==seed)].iloc[0]
        for k,v in score(d.label,d.probability,d.prediction).items():close(v,row[k])
    paths=[p for folder in ('folds','controls','train_only','analysis','tables','cache','checkpoints') for p in (OUT/folder).rglob('*')if p.is_file()]
    write(OUT/'numeric_validation.json',dict(complete=True,checks=dict(protected_files=protected,participants=258,outer_folds=6,
        inner_neural_fits=inner_count,outer_model_records=outer_count,ig_artifacts_validated=ig_count,permutation_evaluations_replayed=permutation_count,
        baseline_oof_matches=True,nested_selection_train_only=True,preprocessing_train_only=True,raw_feature_bank_verified=True,
        robustness_and_missing_task_predictions_replayed=True),
        artifact_sha256={p.relative_to(ROOT).as_posix():sha(p)for p in sorted(paths)},code_sha256=code_hashes))
    print('NUMERICAL VALIDATION PASSED',flush=True)


def final():
    import nbformat
    from PIL import Image
    protected=protect();numeric=json.loads((OUT/'numeric_validation.json').read_text());assert numeric['complete']
    for rel,digest in {**numeric['artifact_sha256'],**numeric['code_sha256']}.items():assert sha(ROOT/rel)==digest,rel
    figures=json.loads((OUT/'figures/manifest.json').read_text())
    assert len(figures)>=10
    for item in figures:
        for rel,digest in item['sources'].items():assert sha(ROOT/rel)==digest,rel
        for ext,digest in item['outputs'].items():
            p=OUT/'figures'/(item['name']+'.'+ext);assert sha(p)==digest
            if ext=='png':
                with Image.open(p)as im:assert min(im.info['dpi'])>=299
    notebook=ROOT/'notebooks/06_kinematic_task_transformer.ipynb';n=nbformat.read(notebook,as_version=4);nbformat.validate(n)
    code=[c for c in n.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in code)
    assert not[o for c in code for o in c.outputs if o.output_type=='error']
    notebook_validation=json.loads((OUT/'notebook_validation.json').read_text());assert len(notebook_validation)==1 and notebook_validation[0]['errors']==0
    test=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_task_transformer.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8');assert test.returncode==0,test.stderr
    for p in (ROOT/'scripts').glob('*task_transformer*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    write(OUT/'validation.json',dict(complete=True,stage='complete',**numeric['checks'],protected_files_final=protected,
       figures=len(figures),dpi=300,unit_tests=9,notebook_executed=True,code_cells=len(code),
       numeric_tables=sum('text/html'in o.get('data',{})for c in code for o in c.outputs),
       notebook_sha256=sha(notebook),report_sha256=sha(OUT/'REPORT_VI.md'),numeric_validation_sha256=sha(OUT/'numeric_validation.json'),
       limitations=['Same cohort and outer folds were inspected in Task 1: research-level adaptive bias remains',
                    'No external or cross-language evaluation; motor invariance is a hypothesis'],
       historical_results_changed=False,cross_language_experiment_run=False))
    print('FINAL VALIDATION COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--numerics-only',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=4):numerics()if args.numerics_only else final()
