"""Recompute nested metrics, replay leakage audits and report all fusion results."""
import json
import time
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from fusion_common import ROOT,OUT,GRID,VARIANTS,METRICS,BRANCH_OPTIONS,option,load_data,signature,sha,write_json
from evaluate import metrics
from evaluate_fusion_ablation import cv,aggregate_inner,choose_scenarios,predict_candidate


def same(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)


def read_stage(tag,ids,tr,te,sig):
    p=OUT/'stages'/f'{tag}.json'; r=json.loads(p.read_text())
    if r['settings_sha256']!=sig or r['train_participants']!=ids[tr].tolist() or r['test_participants']!=ids[te].tolist():
        raise ValueError('Stage participant/provenance mismatch: '+tag)
    if set(r['train_participants'])&set(r['test_participants']): raise ValueError('Stage leakage')
    for b,bank in r['banks'].items():
        if bank['fit_participants']!=ids[tr].tolist(): raise ValueError('Preprocessing leakage')
        for spec,meta in bank['reducers'].items():
            k=meta['n_features']
            if k<1 or k>bank['raw_features']: raise ValueError('Invalid dimensions')
            if spec.startswith('pca') and (k>len(tr)-1 or meta['explained_variance']>1+1e-6):
                raise ValueError('Invalid training rank/variance')
            if spec.startswith('anova'):
                ix=meta['selected_original_indices']
                if len(set(ix))!=k or min(ix)<0 or max(ix)>=bank['raw_features']: raise ValueError('Invalid selector')
    pred=OUT/'stages'/f'{tag}.npz'
    if sha(pred)!=r['pred_sha256']: raise ValueError('Stage predictions changed')
    if 'checkpoint' in r and sha(OUT/r['checkpoint'])!=r['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    with np.load(pred) as z: arrays={k:z[k] for k in z.files}
    for a in arrays.values():
        if a.shape!=(len(te),2) or not np.isfinite(a).all() or not np.isin(a[:,0],[0,1]).all() or (a[:,1]<0).any() or (a[:,1]>1).any():
            raise ValueError('Invalid prediction values')
    return arrays,r


def summarize(frame,extra):
    row=dict(extra,**metrics(frame.label,frame.prediction,frame.probability))
    for m in METRICS:
        vals=[metrics(d.label,d.prediction,d.probability)[m] for _,d in frame.groupby('fold')]
        row['mean_'+m]=float(np.mean(vals)); row['std_'+m]=float(np.std(vals,ddof=1))
    return row


def validate_fold(f,arrays,ids,y,folds,splits,sig):
    tr,te=splits[f]; train_ids=ids[tr]; train_y=y[tr]
    s=json.loads((OUT/'selections'/f'fold_{f}.json').read_text())
    r=json.loads((OUT/'folds'/f'fold_{f}.json').read_text())
    if (s['train_participants']!=train_ids.tolist() or r['train_participants']!=train_ids.tolist()
        or r['test_participants']!=ids[te].tolist() or r['test_indices']!=te.tolist()
        or s['settings_sha256']!=sig or r['settings_sha256']!=sig): raise ValueError('Outer identity changed')
    ip=OUT/'selections'/f'fold_{f}_inner_predictions.npz'
    bp=OUT/'selections'/f'fold_{f}_base_crossfit.npz'
    if sha(ip)!=s['inner_predictions_sha256'] or sha(bp)!=s['base_crossfit_sha256']: raise ValueError('Inner predictions changed')
    with np.load(ip) as z: inner={k:z[k] for k in z.files}
    with np.load(bp) as z: base={k:z[k] for k in z.files}
    if set(inner)!=set(GRID): raise ValueError('Missing inner candidates')
    scores=[]; costs=[]
    for k,(itr,iva) in enumerate(cv(train_ids,train_y)):
        a=s['inner_audits'][k]
        if a['train_participants']!=train_ids[itr].tolist() or a['validation_participants']!=train_ids[iva].tolist() or a['combiner_fit_participants']!=train_ids[itr].tolist():
            raise ValueError('Inner combiner leakage')
        if (set(a['train_participants'])|set(a['validation_participants']))&set(ids[te]): raise ValueError('Outer-test leakage')
        validation,record=read_stage(a['stage'],train_ids,itr,iva,sig)
        costs.extend(dict(fold=f,stage=a['stage'],**c) for c in record['costs'])
        for key in base: same(base[key][iva],validation[key])
        cross={key:np.empty((len(itr),2)) for key in base}
        for j,(ct,cvv) in enumerate(cv(train_ids[itr],train_y[itr])):
            tag=a['crossfit_stages'][j]
            pred,rec=read_stage(tag,train_ids[itr],ct,cvv,sig)
            costs.extend(dict(fold=f,stage=tag,**c) for c in rec['costs'])
            for key in cross: cross[key][cvv]=pred[key]
        for name,c in GRID.items():
            if c['method'] in ('original','compact'):
                expected=validation[name]; d=record['dimensions'][name]
            else:
                expected,_=predict_candidate(c,validation,cross,train_y[itr])
                d=sum(record['dimensions'][o] for o in c['options'])
            same(inner[name][iva],expected)
            scores.append(dict(candidate=name,inner_fold=k,n_features=d,
                               **metrics(train_y[iva],expected[:,0].astype(int),expected[:,1])))
    original_scores=pd.DataFrame(s['inner_scores']).sort_values(['candidate','inner_fold']).reset_index(drop=True)
    recomputed=pd.DataFrame(scores).sort_values(['candidate','inner_fold']).reset_index(drop=True)
    if not original_scores[['candidate','inner_fold']].equals(recomputed[['candidate','inner_fold']]): raise ValueError('Inner grid incomplete')
    same(original_scores[['n_features',*METRICS]],recomputed[['n_features',*METRICS]])
    ranking=aggregate_inner(scores); chosen=choose_scenarios(ranking)
    if chosen!=s['selections'] or set(chosen)!=set(r['results']): raise ValueError('Selection not derived from inner CV')
    validation,outer_stage=read_stage(f'outer_{f}',ids,tr,te,sig)
    costs.extend(dict(fold=f,stage=f'outer_{f}',**c) for c in outer_stage['costs'])
    checkpoint=joblib.load(OUT/outer_stage['checkpoint'])
    if checkpoint['train_participants']!=train_ids.tolist() or checkpoint['settings_sha256']!=sig: raise ValueError('Checkpoint training identity')
    projected={b:bank.transform_many(arrays[b][te]) for b,bank in checkpoint['banks'].items()}
    for b,bank in checkpoint['banks'].items():
        if bank.fit_ids_!=train_ids.tolist(): raise ValueError('Checkpoint bank leakage')
    actual_dimensions={}
    for key,pipe in checkpoint['models'].items():
        if key in GRID:
            c=GRID[key]
            if c['method']=='original': X=np.concatenate([arrays[b][te] for b in c['branches']],axis=1)
            else: X=np.concatenate([projected[b][spec] for b,spec in zip(c['branches'],c['specs'])],axis=1)
        else:
            b,spec,clf=key.split('__'); X=projected[b][spec]
        actual=np.column_stack([pipe.predict(X),pipe.predict_proba(X)[:,1]])
        same(actual,validation[key])
        actual_dimensions[key]=(int(pipe[1].get_support().sum())
                                if key in GRID and GRID[key]['method']=='original' else X.shape[1])
    if sha(OUT/r['combiner_checkpoint'])!=r['combiner_checkpoint_sha256']: raise ValueError('Combiner checkpoint hash')
    cp=joblib.load(OUT/r['combiner_checkpoint'])
    if cp['train_participants']!=train_ids.tolist(): raise ValueError('Combiner checkpoint IDs')
    rows=[]; selected=[]; fold_scores=[]
    for scenario,row in chosen.items():
        name=row['candidate']; c=GRID[name]; result=r['results'][scenario]
        if result['candidate']!=name: raise ValueError('Outer chose different candidate')
        if c['method'] in ('original','compact'): expected=validation[name]
        else:
            expected,model=predict_candidate(c,validation,base,train_y)
            if model is not None:
                if result['combiner_fit_participants']!=train_ids.tolist(): raise ValueError('Outer combiner leakage')
                same(model.weights_,result['combiner']['weights'])
                Q=np.column_stack([validation[o][:,1] for o in c['options']])
                same(cp['models'][scenario].predict_proba(Q)[:,1],expected[:,1])
        same(result['prediction'],expected[:,0]); same(result['probability'],expected[:,1])
        score=metrics(y[te],expected[:,0].astype(int),expected[:,1])
        for m,v in score.items(): same(result['metrics'][m],v)
        rows.extend(dict(scenario=scenario,participant_id=ids[i],label=int(y[i]),fold=f,
                         prediction=int(expected[j,0]),probability=float(expected[j,1])) for j,i in enumerate(te))
        actual_width=(actual_dimensions[name] if c['method'] in ('original','compact')
                      else sum(actual_dimensions[o] for o in c['options']))
        raw_width=sum(arrays[b].shape[1] for b in c['branches'])
        fold_scores.append(dict(scenario=scenario,fold=f,candidate=name,n_features=actual_width,
                                n_features_pipeline=result['n_features'],n_features_raw=raw_width,**score))
        selected.append(dict(scenario=scenario,fold=f,candidate=name,method=c['method'],variant=c['variant'],
                            n_features=result['n_features'],inner_f1_macro=row['mean_f1_macro'],inner_accuracy=row['mean_accuracy'],
                            fallback_to_kinematic=name=='K_only',combiner=json.dumps(result.get('combiner',{}))))
    print('VALIDATED OUTER',f,flush=True)
    return rows,selected,fold_scores,costs


def markdown(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    sig=signature(); arrays,ids,y,folds,splits=load_data()
    expected={'K':1036,'M':14336,'O':10752,'G':75264,'A':10752,'C':25088}
    for branch,width in expected.items():
        if arrays[branch].shape!=(258,width): raise ValueError('Feature cache dimensions changed: '+branch)
    if len(list((OUT/'folds').glob('fold_*.json')))!=6: raise ValueError('All six outer folds must finish')
    began=time.perf_counter(); rows=[]; selected=[]; fold_scores=[]; costs=[]
    for f in range(6):
        a,b,c,d=validate_fold(f,arrays,ids,y,folds,splits,sig)
        rows+=a; selected+=b; fold_scores+=c; costs+=d
    d=pd.DataFrame(rows); fs=pd.DataFrame(fold_scores); sel=pd.DataFrame(selected)
    summary=[]; base=d[d.scenario=='Kinematic_LightGBM'].sort_values('participant_id')
    original=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv').sort_values('participant_id')
    if not np.array_equal(base.participant_id,original.participant_id): raise ValueError('Baseline IDs changed')
    same(base.prediction,original.prediction); same(base.probability,original.probability)
    baseline=metrics(base.label,base.prediction,base.probability)
    for scenario,g in d.groupby('scenario',sort=False):
        if len(g)!=258 or g.participant_id.nunique()!=258: raise ValueError('OOF coverage not exactly 258')
        extra=dict(scenario=scenario,mean_features=float(fs[fs.scenario==scenario].n_features.mean()),
                   mean_raw_features=float(fs[fs.scenario==scenario].n_features_raw.mean()))
        row=summarize(g,extra)
        for m in METRICS: row['delta_oof_'+m+'_pp']=100*(row[m]-baseline[m])
        summary.append(row)
        dest=OUT/'oof'/f'{scenario}.csv'; dest.parent.mkdir(exist_ok=True)
        g.sort_values('participant_id').to_csv(dest,index=False)
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/'nested_summary.csv',index=False); fs.to_csv(OUT/'fold_scores.csv',index=False)
    d.to_csv(OUT/'nested_oof.csv',index=False); sel.to_csv(OUT/'selected_by_fold.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'compute_costs.csv',index=False)
    sel.groupby(['scenario','candidate'],sort=False).size().reset_index(name='n_folds').to_csv(OUT/'selection_frequency.csv',index=False)
    history=[]
    for variant in ['KM','KG','KC','KMG']:
        name={'KM':'Kinematic+MOMENT','KG':'Kinematic+GAF','KC':'Kinematic+CWT','KMG':'Kinematic+MOMENT+GAF'}[variant]
        for p in (ROOT/'results/cv').glob(name+'__*.oof.csv'):
            old=pd.read_csv(p)
            if not np.array_equal(old.participant_id,ids) or not np.array_equal(old.label,y) or not np.array_equal(old.fold,folds.fold):
                raise ValueError('Historical OOF alignment')
            history.append(summarize(old,dict(variant=variant,classifier=p.name.split('__')[1].split('.')[0],selection='historical fixed; not nested tuned')))
    history=pd.DataFrame(history); history.to_csv(OUT/'historical_early_summary.csv',index=False)
    cases=[]; confidence=[]
    for scenario,g in d.groupby('scenario',sort=False):
        g=g.sort_values('participant_id')
        good=(g.prediction.to_numpy()==g.label.to_numpy()); oldgood=(base.prediction.to_numpy()==base.label.to_numpy())
        for j,row in enumerate(g.itertuples()):
            cases.append(dict(scenario=scenario,participant_id=row.participant_id,fold=row.fold,label=row.label,
                     kinematic_prediction=int(base.prediction.iloc[j]),fusion_prediction=row.prediction,
                     kinematic_probability=float(base.probability.iloc[j]),fusion_probability=row.probability,
                     corrected=bool(good[j] and not oldgood[j]),harmed=bool(oldgood[j] and not good[j])))
        confidence.append(dict(scenario=scenario,corrected=int((good&~oldgood).sum()),harmed=int((oldgood&~good).sum()),
                               same_correct=int((oldgood&good).sum()),same_wrong=int((~oldgood&~good).sum())))
    cases=pd.DataFrame(cases); cases.to_csv(OUT/'prediction_changes.csv',index=False)
    changes=pd.DataFrame(confidence); changes.to_csv(OUT/'correction_harm.csv',index=False)
    buckets=[]
    for scenario,g in cases.groupby('scenario',sort=False):
        distance=(g.kinematic_probability-.5).abs()
        for lo,hi in zip([0,.1,.2,.3,.4],[.1,.2,.3,.4,.5]):
            t=g[(distance>=lo)&((distance<hi) if hi<.5 else (distance<=hi))]
            buckets.append(dict(scenario=scenario,confidence_low=lo,confidence_high=hi,n=len(t),
                                corrected=int(t.corrected.sum()),harmed=int(t.harmed.sum())))
    pd.DataFrame(buckets).to_csv(OUT/'confidence_ablation.csv',index=False)
    ablations=[]; index=summary.set_index('scenario')
    for method in ('original','compact','late','adaptive'):
        pairs=[('KM','Kinematic_LightGBM','M'),('KG','Kinematic_LightGBM','G'),('KC','Kinematic_LightGBM','C'),
               ('KMG',method+'__KM','G given KM'),('KMG',method+'__KG','M given KG')]
        if method!='original': pairs += [('KG',method+'__KG_adapter','G concat vs adapter'),('KMG',method+'__KMG_adapter','G concat vs adapter given M')]
        if method=='late': pairs += [(v,'equal__'+v,'learned weight selection vs equal') for v in VARIANTS]
        for variant,reference,effect in pairs:
            name=method+'__'+variant
            row=dict(method=method,scenario=name,reference=reference,effect=effect)
            for m in METRICS:
                row['delta_oof_'+m+'_pp']=100*(index.loc[name,m]-index.loc[reference,m])
                row['delta_mean_'+m+'_pp']=100*(index.loc[name,'mean_'+m]-index.loc[reference,'mean_'+m])
            ablations.append(row)
    pd.DataFrame(ablations).to_csv(OUT/'ablation_comparison.csv',index=False)
    old=json.loads((OUT/'original_artifacts.json').read_text())
    for p,h in old.items():
        if sha(ROOT/p)!=h: raise ValueError('Old artifact modified: '+p)
    write_report(summary,history,changes,sel,len(old))
    numeric={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*')
             if p.is_file() and p.suffix in ('.csv','.npz','.json','.md') and p.name not in ('validation.json','notebook_validation.json','test_validation.json')
             and not p.is_relative_to(OUT/'figures')}
    write_json(OUT/'validation.json',dict(complete=False,stage='numerical_validation_passed',numerical_validation_passed=True,
        n_subjects=258,seed=42,outer_folds=6,inner_folds=3,meta_crossfit_folds=3,
        candidates=len(GRID),scenarios=len(summary),outer_records=6,stages=len(list((OUT/'stages').glob('*.json'))),
        original_artifacts_unchanged=len(old),baseline_oof_exact=True,inner_predictions_replayed=True,
        outer_checkpoints_replayed=True,participant_leakage_checks_passed=True,
        numerical_artifact_sha256=numeric,validation_seconds=time.perf_counter()-began))
    print('NUMERICAL VALIDATION PASSED',len(summary),'scenarios',flush=True)


def write_report(summary,history,changes,selected,n_old):
    index=summary.set_index('scenario'); base=index.loc['Kinematic_LightGBM']
    core=summary[summary.scenario.isin(['Kinematic_LightGBM','all_selected']+
         [m+'__'+v for m in ('original','compact','late','adaptive') for v in ('KM','KG','KC','KMG')])].copy()
    display=core[['scenario','mean_raw_features','mean_features']].copy()
    for m in METRICS:
        display['OOF_'+m]=core[m]*100
        display['fold_'+m+' mean±SD']=[f'{a*100:.2f} ± {b*100:.2f}' for a,b in zip(core['mean_'+m],core['std_'+m])]
    display['Δ OOF F1 pp']=core.delta_oof_f1_macro_pp
    best=core[~core.scenario.isin(['Kinematic_LightGBM','all_selected'])].sort_values('mean_f1_macro',ascending=False).iloc[0]
    improved=bool(((core.mean_f1_macro>base.mean_f1_macro)&(core.accuracy>=base.accuracy)
                   & ~core.scenario.isin(['Kinematic_LightGBM','all_selected'])).any())
    policy=index.loc['all_selected']
    policy_improved=policy.mean_f1_macro>base.mean_f1_macro and policy.accuracy>=base.accuracy
    recommendations=('Quy trình chọn trong Train có cải thiện mô tả; giữ checkpoint nghiên cứu và kiểm tra trên cohort ngoài trước khi thay baseline.'
                     if policy_improved else 'Giữ Kinematic + LightGBM làm mô hình chính. Quy trình mở rộng chọn Fusion chưa đồng thời tăng mean F1 và giữ Accuracy OOF; không thay baseline bằng một hàng thắng hậu nghiệm.')
    best_by_method=[]
    for method in ('original','compact','late','adaptive'):
        winner=core[core.scenario.str.startswith(method+'__')].sort_values('mean_f1_macro',ascending=False).iloc[0]
        change=changes.set_index('scenario').loc[winner.scenario]
        best_by_method.append(dict(method=method,scenario=winner.scenario,OOF_Accuracy=winner.accuracy*100,
              OOF_F1=winner.f1_macro*100,mean_F1=winner.mean_f1_macro*100,OOF_AUC=winner.auc*100,
              delta_F1_pp=winner.delta_oof_f1_macro_pp,corrected=int(change.corrected),harmed=int(change.harmed)))
    branch_deltas=pd.read_csv(OUT/'ablation_comparison.csv')
    branch_deltas=branch_deltas[branch_deltas.effect.isin(['M','G','C','G given KM','M given KG'])][
        ['method','effect','delta_oof_accuracy_pp','delta_oof_f1_macro_pp','delta_oof_auc_pp']]
    adapter=summary[summary.scenario.str.contains('adapter')][['scenario','mean_features','accuracy','f1_macro','auc','balanced_accuracy']].copy()
    adapter.loc[:,METRICS]=adapter[METRICS]*100
    h=history.sort_values('mean_f1_macro',ascending=False).groupby('variant',sort=False).head(1)[['variant','classifier','accuracy','f1_macro','auc']].copy()
    h.loc[:,['accuracy','f1_macro','auc']]=h[['accuracy','f1_macro','auc']]*100
    cost=pd.read_csv(OUT/'compute_costs.csv')
    seconds=float(cost.seconds.sum()); checkpoint_mb=sum(p.stat().st_size for p in (OUT/'checkpoints').glob('*.joblib'))/1024**2
    verdict=('Có cải thiện mô tả ở cả F1 trung bình fold và Accuracy OOF so với Kinematic.' if improved else
             'Chưa có cấu hình Fusion đáp ứng đồng thời F1 trung bình fold cao hơn và Accuracy OOF không giảm so với Kinematic.')
    report=f'''# Multimodal Fusion — Nested CV và Ablation Study

## Kết quả và trạng thái

**{verdict}** Cấu hình Fusion có mean F1 cao nhất trong bốn phương pháp/tổ hợp chính là
`{best.scenario}`: OOF Accuracy {best.accuracy*100:.2f}%, F1 macro {best.f1_macro*100:.2f}%,
ROC-AUC {best.auc*100:.2f}%; Δ OOF F1 {best.delta_oof_f1_macro_pp:+.2f} điểm phần trăm.
Đây là hàng đứng đầu sau khi xem kết quả, có thiên lệch chọn hàng thắng. `all_selected` là quy trình
chọn giữa các phương pháp/tổ hợp chính hoàn toàn bằng inner CV, cần đọc cùng hàng này.

Baseline chính **Kinematic nguyên chiều + LightGBM**, giữ nguyên tham số cũ:
OOF Accuracy **{base.accuracy*100:.2f}%**, F1 macro **{base.f1_macro*100:.2f}%**,
ROC-AUC **{base.auc*100:.2f}%**. Dự đoán và xác suất tái fit khớp baseline cũ trong tolerance 1e-12.
Kết quả mới không thay thế baseline. Chỉ `validation.json complete=true` xác nhận hoàn tất sau
kiểm tra notebook, hình, checkpoint, OOF, nguồn dữ liệu và tests.

## Thiết kế cố định trước khi chạy

258 participant duy nhất, nhãn `group`, outer Stratified 6-Fold với assignments cũ, seed 42;
inner Stratified 3-Fold chọn theo mean F1 macro, sau đó Accuracy, ROC-AUC, ít chiều và ID cấu hình.
Không chọn threshold; giữ ngưỡng xác suất 0.5 cho bộ kết hợp. Mô hình đơn và Early dùng predict()
nguyên bản (SVM predict có thể khác threshold predict_proba). Không chọn dựa outer Test.

Các tổ hợp chính: KM=Kinematic+MOMENT, KG=Kinematic+GAF, KC=Kinematic+CWT,
KMG=Kinematic+MOMENT+GAF. Hai đối chứng KG_adapter/KMG_adapter dùng GAF group_adapter.
Không mở rộng sang tổ hợp khác. Tất cả cache embedding label-free giữ nguyên; GPU không được sử dụng,
không trích xuất hoặc fine-tune encoder, không huấn luyện trên toàn cohort để tạo OOF.

- **Original Early:** embedding gốc MOMENT/GAF/CWT, nối vector và pipeline cũ;
  inner CV chọn LogisticRegression/SVM/LightGBM. Tách riêng bảng lịch sử bảy classifier cũ và Dummy đối chứng.
- **Compact Early:** Kinematic nguyên chiều; giảm chiều riêng từng nhánh, rồi nối và chọn một trong
  ba classifier trên. MOMENT ANOVA-64 hoặc PCA-99%; GAF channel_concat (và adapter đối chứng)
  ANOVA-64/256 hoặc PCA-95%; CWT embedding gốc, none hoặc PCA-99% (none được phép thắng).
- **Late:** classifier riêng: Kinematic none/LightGBM cố định; MOMENT ANOVA-64/LR hoặc PCA-99%/SVM;
  GAF ANOVA-256/LightGBM, ANOVA-64/SVM hoặc PCA-95%/LR; CWT none/LR hoặc PCA-99%/LR.
  Chọn equal pooling hoặc trọng số simplex không âm tổng bằng 1 học bằng log-loss trên cross-fitted
  Train probabilities, regularization về uniform λ=0 hoặc 0.1. `equal__*` là đối chứng chỉ cho uniform.
- **Adaptive/Selective:** giữ Kinematic làm mô hình chính. Với |pK−0.5|≤τ (τ=0.1/0.25/0.5),
  p=pK+α×(Σw_aux p_aux−pK), α=0.25/0.5, simplex w_aux học từ cross-fitted probabilities,
  λ=0.1. Ngoài vùng bất định, giữ nguyên pK. Inner CV chỉ chấp nhận ứng viên nếu mean F1 tăng
  nghiêm ngặt **và** mean Accuracy không giảm; nếu không, trả về đúng Kinematic. Đây là quy tắc chọn
  trên Train, **không bảo đảm** không suy giảm trên outer Test. Gate không giảm chi phí encoder
  trong thí nghiệm vì mọi embedding đã cache và mọi nhánh đều được dự đoán.

Lưới có **{len(GRID)} ứng viên**, gồm cả đơn nhánh và fallback; `{len(summary)}` quy trình nested,
6 outer jobs, 78 stage fit (6×[3 inner chính + 9 cross-fit inner Train + 1 outer]).
`settings.json` giữ đầy đủ classifier params, grid, seed, versions và SHA-256 code/input.
Lưới nhỏ dựa trên nghiên cứu trước, không quét mọi reducer/classifier đã có.

## Chống data leakage và checkpoint

Mỗi branch bank median imputer → variance filter → StandardScaler chỉ fit trên Train của split.
PCA full SVD và ANOVA dùng đúng Train đó; Kinematic không giảm chiều. PCA/selector không được
chép từ nghiên cứu trước hoặc từ outer Train xuống inner Train. Chia sẻ tính toán chỉ giữa các
ứng viên có cùng tập Train. PCA không whitening và không rescale component sau projection.

Để đánh giá một cấu hình Late/Adaptive trên inner Validation, trước hết cross-fit **3-fold riêng
trong inner Train** cho từng base model, refit cả preprocessing/reducer. Bộ kết hợp chỉ học từ
các dự đoán này và nhãn inner Train, sau đó nhận xác suất Validation của base model fit đủ inner Train.
Vì vậy nhãn của một inner Validation không đi vào preprocessing, base model hay combiner của split đó.
Sau khi chọn bằng ba inner Validation, outer combiner fit từ cross-fitted base predictions của
215 outer Train (ba inner chính); base models refit đủ 215 người rồi dự đoán 43 outer Test.

Validator dựng lại split/coverage, kiểm tra fit IDs của từng bank và combiner, tái tính toàn bộ
inner scores/selection từ predictions, refit combiner từ đúng cross-fit cache, nạp và chạy lại
outer checkpoints, tái tính metrics từ OOF. Không dùng mô hình train full cohort làm bằng chứng OOF.
Lưu 6 checkpoint base/bank/selected Early và 6 checkpoint combiner trong `checkpoints/`.
Các joblib ở máy (gitignored để tránh repository quá lớn); script tái tạo dùng cache có sẵn.

## Bảng so sánh chính

Đơn vị **%**, Δ là **điểm phần trăm**. OOF gộp và mean±SD sáu fold được tách cột;
SD giữa các fold có tập Train chồng lấp **không phải CI**. Bảng đầy đủ ở `nested_summary.csv`.
`mean_raw_features` là chiều cache của các nhánh được chọn; `mean_features` là tổng chiều thực
sự vào classifier sau preprocessing/reducer. Kinematic vẫn dùng đủ 1.036 đặc trưng gốc,
chỉ loại cột hằng theo Train như baseline (trung bình còn 1.027,67 cột), không PCA/feature selection.
Nếu Adaptive fallback về Kinematic, số chiều của hàng đó chỉ tính mô hình đang đóng góp;
chi phí thực nghiệm vẫn bao gồm tính xác suất các nhánh phụ để chọn cấu hình.

{markdown(display)}

## Đối chứng GAF chi phí thấp

{markdown(adapter)}

`ablation_comparison.csv` so sánh channel_concat với adapter trong cùng phương pháp, và
KMG với KM/KG. Đây là ablation **quy trình**: từng tổ hợp được chọn reducer/classifier/weights
riêng trong Train, không phải hiệu ứng nhân quả thuần của một tọa độ. Không gán mọi cải thiện
cho thêm một nhánh khi cấu hình học thay đổi.

## So với Early Fusion đã có

Bảng sau chỉ tóm tắt classifier lịch sử có mean F1 cao nhất mỗi tổ hợp (bảy classifier và Dummy đối chứng);
**đã chọn theo outer grid cũ**, có selection bias, không xem là nested estimate. Toàn bộ các hàng
lịch sử giữ ở `historical_early_summary.csv`. Original Early mới chỉ chọn trong ba classifier
đại diện bằng inner CV, nên khác pool và khác quy trình chọn. Baseline chính vẫn là LightGBM
Kinematic cố định, không đổi sang classifier do inner CV chọn.

{markdown(h)}

## Nhánh nào sửa đúng, nhánh nào gây hại

{markdown(changes[changes.scenario.isin(core.scenario)].copy())}

`prediction_changes.csv` lưu từng người (baseline/fusion probabilities, corrected/harmed);
`confidence_ablation.csv` chia độ tự tin Kinematic thành năm khoảng định trước. Những trường hợp
này là mô tả hậu nghiệm, không dùng để huấn luyện gate hoặc chỉnh threshold sau khi xem Test.
`selected_by_fold.csv` ghi fallback và tham số combiner; `selection_frequency.csv` mô tả độ ổn định.
Xem dấu và độ lớn chênh lệch từng fold trong notebook trước khi quyết định giữ một nhánh.

### Phương pháp đáng giữ và giới hạn của mức tăng

Hàng mean F1 cao nhất mỗi phương pháp, **xếp hạng mô tả sau outer CV**, không dùng các hàng này
để chọn lại tham số hoặc tạo OOF mới:

{markdown(pd.DataFrame(best_by_method))}

Quy trình chọn tổng thể `all_selected`: Accuracy OOF **{policy.accuracy*100:.2f}%**
({policy.delta_oof_accuracy_pp:+.2f} điểm), F1 OOF **{policy.f1_macro*100:.2f}%**
({policy.delta_oof_f1_macro_pp:+.2f} điểm), mean F1 **{policy.mean_f1_macro*100:.2f}%**
so với {base.mean_f1_macro*100:.2f}% của Kinematic. {recommendations}
Một người được dự đoán đúng thêm tương ứng khoảng 0.388 điểm Accuracy trên 258 người;
không xem vài trường hợp sửa đúng là bằng chứng chắc chắn của tổng quát hóa.

Tác động thêm từng nhánh (điểm phần trăm OOF):

{markdown(branch_deltas)}

Các nhánh phụ có thể bổ sung khi lỗi của chúng khác Kinematic, nhưng cũng có thể kéo xác suất
của một dự đoán đúng qua ngưỡng 0.5. Bảng corrected/harmed đo trực tiếp hai chiều tác động này.
Chênh lệch GAF concat/adapter cần cân nhắc chi phí encode ~7 lần; CWT không được ưu tiên chỉ
vì có một fold tốt. Fallback nhiều lần cho biết inner CV không tìm thấy lợi ích theo safety rule,
không chứng minh nhánh đó hoàn toàn không chứa thông tin. Không suy nguyên nhân sinh học từ ablation.

## Chi phí và ý nghĩa thực tế

Tổng thời gian các thành phần fit/transform đã đo: **{seconds/60:.2f} phút** (tổng workload, có
chạy hai outer jobs đồng thời; **không phải wall time**). Checkpoint nén **{checkpoint_mb:.1f} MiB**.
`compute_costs.csv` tách stage/bank/classifier và thời gian predict, không bao gồm mọi thao tác I/O,
meta fitting hoặc trích xuất encoder. Late/Adaptive dùng thêm cross-fitting nên chi phí train lớn hơn
một Early cố định; tổng thời gian tìm kiếm không đồng nghĩa latency triển khai.
GAF channel_concat vẫn cần bảy lần encode kênh; benchmark trước khoảng 7.21× embedding gốc,
adapter khoảng 0.99×. Giảm chiều chỉ giảm classifier input, không loại bỏ chi phí encoder.
Không có benchmark end-to-end Fusion trên tín hiệu thô trong nhiệm vụ này.

Ưu tiên chọn phương pháp đơn giản nếu chỉ tăng rất ít. Quy trình `all_selected` cho phép trả về
Kinematic qua safety rule, nhằm kiểm tra liệu việc mở pool Fusion có lợi khi lựa chọn hoàn toàn
trong Train. Không dùng hàng thắng outer để triển khai tự động hoặc ghi đè baseline.

Thiết kế đã tham khảo kết quả trên **cùng cohort 258 người**, nên nested CV đợt này không phải một
tập xác nhận mới độc lập. Không suy ra ý nghĩa thống kê hay hiệu quả chẩn đoán lâm sàng từ sáu fold;
cần external holdout hoặc cohort khác để xác nhận mức tăng nhỏ. Không thực hiện XAI.

## Artifacts và chạy lại

Giữ nguyên {n_old} artifact cũ bằng SHA-256. Kết quả mới chỉ ở `results/fusion_ablation/`;
notebook mới `notebooks/08_fusion_ablation.ipynb`, tests `tests/test_fusion.py`.
`stages/` lưu cross-fit probabilities, fit IDs, selected features, PCA variance và chi phí;
`selections/` lưu toàn bộ inner candidates/scores/predictions; `folds/`, `oof/`,
`fold_scores.csv`, `nested_oof.csv`, PNG/SVG và manifest phục vụ kiểm chứng.
`code/` giữ bản sao mã thực nghiệm và phân tích để review độc lập.

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/evaluate_fusion_ablation.py --workers 2
.venv/Scripts/python.exe -X utf8 -B -u scripts/report_fusion_ablation.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_fusion_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 08_fusion_ablation.ipynb --validation-output results/fusion_ablation/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_fusion_ablation.py
```

Evaluation có checkpoint và từ chối trộn input/code/protocol đã đổi. Tạo lại báo cáo/notebook
không thay OOF. Cơ sở kỹ thuật: [StackingClassifier — cross-validated training of final estimator](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.StackingClassifier.html),
[cross_val_predict](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.cross_val_predict.html),
[nested CV](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
'''
    (OUT/'REPORT_VI.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
