"""Audit OOF attributions, aggregate feature/task importance and write XAI report."""
import json
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata,spearmanr
from threadpoolctl import threadpool_limits
from xai_common import ROOT,OUT,FUSION,SCENARIOS,N_REPEATS,load_data,initialize,load_models,sha,write_json
from diagramo.explain import feature_mapping,aggregate_shap,validate_additivity,permute_block,fusion_mechanism,outcomes,exact_partition_explainers
from evaluate import metrics

warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names')


def load_attributions():
    output={}
    for f in range(6):
        with np.load(OUT/'attributions'/f'kinematic_fold_{f}.npz') as z:
            if not output:
                for k in z.files:
                    if k!='kept_indices': output[k]=np.empty((258,*z[k].shape[1:]),dtype=z[k].dtype)
            indices=z['indices']
            for k in output: output[k][indices]=z[k]
    return output


def check_fold(f,arrays,ids,y,splits,mapping,sig):
    tr,te=splits[f]; r=json.loads((OUT/'folds'/f'fold_{f}.json').read_text())
    if r['settings_sha256']!=sig or r['train_participants']!=ids[tr].tolist() or r['test_participants']!=ids[te].tolist():
        raise ValueError('XAI fold identity/provenance mismatch')
    if r['kinematic_fit_participants']!=ids[tr].tolist() or set(r['train_participants'])&set(r['test_participants']):
        raise ValueError('XAI model/attribution leakage')
    expected=np.sort(np.random.default_rng(42).choice(tr,size=100,replace=False))
    if r['background_indices']!=expected.tolist() or r['background_participants']!=ids[expected].tolist():
        raise ValueError('Background is not the fixed training-only sample')
    for p,h in r['outputs_sha256'].items():
        if sha(OUT/p)!=h: raise ValueError('XAI numeric file changed: '+p)
    bundle,combiners,record,stage,reconstructed=load_models(f,arrays,ids,y,tr)
    kbank=bundle['banks']['K']; pipe=bundle['models']['K__none__LightGBM']; model=pipe[-1]
    A=kbank.transform_many(arrays['K'][te])['none']; prob=pipe.predict_proba(A)[:,1]; pred=pipe.predict(A)
    B=kbank.transform_many(arrays['K'][expected])['none']
    raw_explainer,_,encode,partition_audit=exact_partition_explainers(model,B)
    if partition_audit!=r['partition_encoding']: raise ValueError('Partition encoding does not match frozen tree thresholds')
    for T in (A,B):
        np.testing.assert_allclose(raw_explainer.model.predict(encode(T)),model.predict(T,raw_score=True),atol=1e-12)
    if any(v>1e-12 for v in r['partition_prediction_equivalence'].values()): raise ValueError('Explainer changed predictions')
    original=pd.read_csv(ROOT/'results/cv/kinematic__LightGBM.oof.csv').iloc[te]
    np.testing.assert_allclose(prob,original.probability,atol=1e-12,rtol=1e-10)
    np.testing.assert_array_equal(pred,original.prediction)
    with np.load(OUT/'attributions'/f'kinematic_fold_{f}.npz') as z:
        np.testing.assert_array_equal(z['indices'],te); np.testing.assert_array_equal(z['participant_ids'],ids[te])
        np.testing.assert_array_equal(z['feature_values'],arrays['K'][te])
        keep=kbank.pre_[1].get_support(indices=True); np.testing.assert_array_equal(z['kept_indices'],keep)
        removed=np.setdiff1d(np.arange(1036),keep)
        margin=model.predict(A,raw_score=True)
        for key,base,out in [('raw_path','base_raw_path',margin),('raw_interventional','base_raw_interventional',margin),
                             ('probability','base_probability',prob)]:
            if z[key].shape!=(43,1036) or (len(removed) and np.any(z[key][:,removed]!=0)): raise ValueError('SHAP alignment/zero padding')
            validate_additivity(z[key],z[base],out)
        np.testing.assert_allclose(z['raw_score'],margin,atol=1e-12)
        np.testing.assert_allclose(expit(z['raw_score']),prob,atol=1e-12)
        for level in ('task','group'):
            _,signed,_=aggregate_shap(z['raw_path'],mapping,level)
            validate_additivity(signed,z['base_raw_path'],margin)
    # Replay every held-out block permutation; score labels are used only here.
    table=pd.read_csv(OUT/'permutation'/f'fold_{f}.csv')
    if len(table)!=26*N_REPEATS: raise ValueError('Permutation coverage incomplete')
    baseline=metrics(y[te],pred,prob)
    with np.load(OUT/'permutation'/f'fold_{f}.npz') as z:
        rng=np.random.default_rng(42+1000*f)
        for b,block in enumerate(r['permutation_blocks']):
            columns=mapping[mapping[block['level']]==block['name']].feature_index.to_numpy()
            if block['feature_indices']!=columns.tolist(): raise ValueError('Permutation feature mapping mismatch')
            for rep in range(N_REPEATS):
                order=rng.permutation(len(te)); np.testing.assert_array_equal(z['permutation_indices'][b,rep],order)
                altered=permute_block(arrays['K'][te],columns,order)
                T=kbank.transform_many(altered)['none']
                pp=pipe.predict_proba(T)[:,1]; yp=pipe.predict(T)
                np.testing.assert_allclose(pp,z['probability'][b,rep],atol=1e-12)
                np.testing.assert_array_equal(yp,z['prediction'][b,rep])
                row=table[(table.level==block['level'])&(table.block==block['name'])&(table['repeat']==rep)]
                if len(row)!=1: raise ValueError('Duplicate permutation scoring row')
                score=metrics(y[te],yp,pp)
                for m in ('accuracy','f1_macro','auc','balanced_accuracy'):
                    np.testing.assert_allclose(row['drop_'+m+'_pp'].iloc[0],100*(baseline[m]-score[m]),atol=1e-10)
    fusion=pd.read_csv(OUT/'folds'/f'fusion_{f}.csv')
    settings=json.loads((FUSION/'settings.json').read_text()); source=pd.read_csv(FUSION/'nested_oof.csv')
    C=bundle['banks']['C'].transform_many(arrays['C'][te])
    for scenario in SCENARIOS:
        frame=fusion[fusion.scenario==scenario].set_index('participant_id').loc[ids[te]]
        c=settings['grid'][record['results'][scenario]['candidate']]; key=c['options'][1]; spec=key.split('__')[1]
        cp=bundle['models'][key]; pc=cp.predict_proba(C[spec])[:,1]
        mechanism=fusion_mechanism(prob,pc,combiners['models'][scenario].metadata(),c['method'])
        for col,expected_ in [('kinematic_probability',prob),('cwt_probability',pc),('fusion_probability',mechanism['probability']),
            ('delta_cwt',mechanism['delta_cwt']),('effective_weight_cwt',mechanism['effective_weight_cwt'])]:
            np.testing.assert_allclose(frame[col],expected_,atol=1e-12)
        old=source[(source.scenario==scenario)&(source.fold==f)].set_index('participant_id').loc[ids[te]]
        np.testing.assert_allclose(frame.fusion_probability,old.probability,atol=1e-12)
        np.testing.assert_array_equal(frame.fusion_prediction,old.prediction)
        np.testing.assert_array_equal(frame.outcome,outcomes(y[te],pred,old.prediction))
        if c['method']=='adaptive': np.testing.assert_array_equal(frame.gate_active.astype(bool),mechanism['gate_active'])
        with np.load(OUT/'attributions'/f'cwt_{scenario}_fold_{f}.npz') as z:
            validate_additivity(z['values'],z['base_value'],cp[-1].decision_function(C[spec]),atol=1e-8)
            np.testing.assert_allclose(expit(z['base_value']+z['values'].sum(1)),pc,atol=1e-12)
    print('XAI VALIDATED OUTER',f,flush=True)


def importance_tables(values,mapping,fold_ids,mode):
    tables={}; fold_tables={}; signed_tables=[]
    for level in ('feature','task','group'):
        rows=[]
        if level=='feature':
            names=mapping.full_name.tolist(); signed=values; mass=np.abs(values)
        else: names,signed,mass=aggregate_shap(values,mapping,level)
        for f in range(6):
            ix=fold_ids==f; scores=mass[ix].mean(0); ranks=rankdata(-scores,method='average')
            top=set(np.argsort(-scores,kind='stable')[:20 if level=='feature' else 3])
            for j,name in enumerate(names):
                n=1 if level=='feature' else int(mapping[level].eq(name).sum())
                rows.append(dict(level=level,name=name,fold=f,n_features=n,sum_mean_abs_shap=float(scores[j]),
                     mean_abs_signed_sum=float(np.abs(signed[ix,j]).mean()),mean_signed_sum=float(signed[ix,j].mean()),
                     per_feature_mean_abs=float(scores[j]/n),rank=float(ranks[j]),in_top_k=j in top,mode=mode))
        frame=pd.DataFrame(rows); fold_tables[level]=frame
        summary=frame.groupby('name',sort=False).agg(n_features=('n_features','first'),
             sum_mean_abs_shap=('sum_mean_abs_shap','mean'),fold_sd_mass=('sum_mean_abs_shap','std'),
             mean_abs_signed_sum=('mean_abs_signed_sum','mean'),mean_signed_sum=('mean_signed_sum','mean'),
             per_feature_mean_abs=('per_feature_mean_abs','mean'),mean_rank=('rank','mean'),sd_rank=('rank','std'),
             top_k_fold_count=('in_top_k','sum')).reset_index()
        summary['attribution_mass_share_percent']=100*summary.sum_mean_abs_shap/summary.sum_mean_abs_shap.sum()
        summary['rank']=rankdata(-summary.sum_mean_abs_shap,method='average')
        if level=='feature': summary=summary.merge(mapping,left_on='name',right_on='full_name',validate='one_to_one')
        summary=summary.sort_values(['rank','name']).reset_index(drop=True); tables[level]=summary
        if level!='feature':
            for j,name in enumerate(names):
                for i in range(258): signed_tables.append(dict(participant_index=i,level=level,name=name,
                    signed_shap=float(signed[i,j]),absolute_mass=float(mass[i,j]),mode=mode))
    return tables,fold_tables,pd.DataFrame(signed_tables)


def stability(fold_tables):
    rows=[]
    for level,frame in fold_tables.items():
        wide=frame.pivot(index='name',columns='fold',values='sum_mean_abs_shap')
        k=20 if level=='feature' else 3
        for a,b in itertools.combinations(range(6),2):
            topa=set(wide[a].nlargest(k).index); topb=set(wide[b].nlargest(k).index)
            rows.append(dict(level=level,fold_a=a,fold_b=b,spearman=float(spearmanr(wide[a],wide[b]).statistic),
                             top_k=k,top_k_jaccard=len(topa&topb)/len(topa|topb)))
    return pd.DataFrame(rows)


def mdtable(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    sig=initialize(); arrays,ids,y,folds,splits=load_data()
    if len(list((OUT/'folds').glob('fold_*.json')))!=6: raise ValueError('All six XAI folds must finish')
    mapping=feature_mapping(json.loads((ROOT/'data/derived/schema.json').read_text()))
    saved=pd.read_csv(OUT/'feature_mapping.csv')
    pd.testing.assert_frame_equal(mapping,saved)
    for f in range(6): check_fold(f,arrays,ids,y,splits,mapping,sig)
    all_=load_attributions(); fold_ids=folds.fold.to_numpy()
    tables,by_fold,signed=importance_tables(all_['raw_path'],mapping,fold_ids,'raw_path')
    raw_iv,_,_=importance_tables(all_['raw_interventional'],mapping,fold_ids,'raw_interventional')
    probability,_,prob_signed=importance_tables(all_['probability'],mapping,fold_ids,'probability')
    for level in tables:
        tables[level].to_csv(OUT/f'{level}_importance.csv',index=False)
        by_fold[level].to_csv(OUT/f'{level}_importance_by_fold.csv',index=False)
        raw_iv[level].to_csv(OUT/f'{level}_importance_interventional_raw.csv',index=False)
        probability[level].to_csv(OUT/f'{level}_importance_probability.csv',index=False)
    tables['feature'].head(30).to_csv(OUT/'top_features.csv',index=False)
    tables['task'].to_csv(OUT/'top_tasks.csv',index=False); tables['group'].to_csv(OUT/'top_feature_groups.csv',index=False)
    contributions=pd.concat([signed,prob_signed],ignore_index=True)
    contributions['participant_id']=ids[contributions.participant_index]
    contributions['fold']=fold_ids[contributions.participant_index]
    contributions.to_csv(OUT/'task_group_participant_contributions.csv',index=False)
    checks=pd.DataFrame(dict(participant_id=ids,fold=fold_ids,label=y,prediction=all_['prediction'],probability=all_['predicted_probability'],
        raw_score=all_['raw_score'],raw_path_reconstructed=all_['base_raw_path']+all_['raw_path'].sum(1),
        raw_interventional_reconstructed=all_['base_raw_interventional']+all_['raw_interventional'].sum(1),
        probability_reconstructed=all_['base_probability']+all_['probability'].sum(1),
        sigmoid_raw_score=expit(all_['raw_score']),interventional_raw_base=all_['base_raw_interventional'],
        sigmoid_interventional_raw_base=expit(all_['base_raw_interventional']),probability_base=all_['base_probability']))
    checks.to_csv(OUT/'output_space_checks.csv',index=False)
    top_rows=[]
    for i in range(258):
        for rank,j in enumerate(np.argsort(-np.abs(all_['raw_path'][i]),kind='stable')[:10],1):
            top_rows.append(dict(participant_id=ids[i],fold=int(fold_ids[i]),rank=rank,feature_index=int(j),
                full_name=mapping.full_name.iloc[j],feature_value=float(arrays['K'][i,j]),
                raw_log_odds_shap=float(all_['raw_path'][i,j]),probability_shap=float(all_['probability'][i,j])))
    pd.DataFrame(top_rows).to_csv(OUT/'participant_top_features.csv',index=False)
    heat=[]
    for (task,group),frame in mapping.groupby(['task','group'],sort=False):
        mass=np.abs(all_['raw_path'][:,frame.feature_index]).sum(axis=1)
        heat.append(dict(task=task,group=group,n_features=len(frame),sum_mean_abs_shap=float(mass.mean()),
                         per_feature_mean_abs=float(mass.mean()/len(frame))))
    pd.DataFrame(heat).to_csv(OUT/'task_group_importance.csv',index=False)
    pairwise=stability(by_fold); pairwise.to_csv(OUT/'fold_stability.csv',index=False)
    stable=pairwise.groupby('level',sort=False).agg(mean_spearman=('spearman','mean'),min_spearman=('spearman','min'),
        max_spearman=('spearman','max'),mean_top_k_jaccard=('top_k_jaccard','mean')).reset_index()
    stable.to_csv(OUT/'fold_stability_summary.csv',index=False)
    task_ranks=by_fold['task'].pivot(index='name',columns='fold',values='rank')
    task_ranks.columns=[f'fold_{f}_rank' for f in task_ranks.columns]
    tables['task'].merge(task_ranks,left_on='name',right_index=True).to_csv(OUT/'task_rank_stability.csv',index=False)
    sensitivity=[]
    for level in tables:
        a=tables[level].set_index('name'); b=raw_iv[level].set_index('name').loc[a.index]; c=probability[level].set_index('name').loc[a.index]
        sensitivity.append(dict(level=level,raw_path_vs_interventional_raw_spearman=float(spearmanr(a.sum_mean_abs_shap,b.sum_mean_abs_shap).statistic),
             interventional_raw_vs_probability_spearman=float(spearmanr(b.sum_mean_abs_shap,c.sum_mean_abs_shap).statistic)))
    pd.DataFrame(sensitivity).to_csv(OUT/'explainer_sensitivity.csv',index=False)
    perm=pd.concat([pd.read_csv(OUT/'permutation'/f'fold_{f}.csv') for f in range(6)],ignore_index=True)
    perm.to_csv(OUT/'permutation_repeats.csv',index=False)
    metrics_=['accuracy','f1_macro','auc','balanced_accuracy']; aggregate={}
    for m in metrics_:
        aggregate['mean_drop_'+m+'_pp']=('drop_'+m+'_pp','mean')
        aggregate['repeat_sd_'+m+'_pp']=('drop_'+m+'_pp','std')
    per_fold=perm.groupby(['fold','level','block'],sort=False).agg(n_features=('n_features','first'),**aggregate).reset_index()
    per_fold.to_csv(OUT/'permutation_fold_scores.csv',index=False)
    aggregate={}
    for m in metrics_:
        aggregate['mean_drop_'+m+'_pp']=('mean_drop_'+m+'_pp','mean')
        aggregate['fold_sd_drop_'+m+'_pp']=('mean_drop_'+m+'_pp','std')
    per_summary=per_fold.groupby(['level','block'],sort=False).agg(n_features=('n_features','first'),**aggregate).reset_index()
    per_summary.to_csv(OUT/'permutation_importance.csv',index=False)
    concordance=[]
    for level in ('task','group'):
        a=tables[level].set_index('name'); b=per_summary[per_summary.level==level].set_index('block').loc[a.index]
        concordance.append(dict(level=level,shap_mass_vs_permutation_f1_spearman=float(spearmanr(a.sum_mean_abs_shap,b.mean_drop_f1_macro_pp).statistic)))
    pd.DataFrame(concordance).to_csv(OUT/'shap_permutation_concordance.csv',index=False)
    fusion=pd.concat([pd.read_csv(OUT/'folds'/f'fusion_{f}.csv') for f in range(6)],ignore_index=True)
    if len(fusion)!=516 or fusion.groupby('scenario').participant_id.nunique().ne(258).any(): raise ValueError('Fusion participant coverage')
    fusion['gate_status']=np.where(~fusion.gate_applicable,'not_applicable',np.where(fusion.gate_active.eq(True),'active','inactive'))
    fusion.to_csv(OUT/'fusion_participants.csv',index=False)
    fusion[fusion.outcome.isin(['corrected','harmed'])].to_csv(OUT/'fusion_corrected_harmed_cases.csv',index=False)
    outcome_summary=fusion.groupby(['scenario','outcome'],sort=False).agg(n=('participant_id','size'),
        mean_kinematic_probability=('kinematic_probability','mean'),mean_cwt_probability=('cwt_probability','mean'),
        mean_fusion_probability=('fusion_probability','mean'),mean_delta_cwt=('delta_cwt','mean'),
        mean_effective_cwt_weight=('effective_weight_cwt','mean')).reset_index()
    outcome_summary.to_csv(OUT/'fusion_outcome_summary.csv',index=False)
    fold_summary=fusion.groupby(['scenario','fold'],sort=False).agg(cwt_reducer=('cwt_reducer','first'),cwt_dimensions=('cwt_dimensions','first'),
        mean_effective_cwt_weight=('effective_weight_cwt','mean'),gate_threshold=('gate_threshold','first'),budget=('budget','first'),
        mean_abs_delta_cwt=('delta_cwt',lambda s:float(np.abs(s).mean()))).reset_index()
    fold_summary.to_csv(OUT/'fusion_fold_mechanisms.csv',index=False)
    bins=[]
    for scenario,frame in fusion.groupby('scenario',sort=False):
        for lo,hi in zip([0,.1,.2,.3,.4],[.1,.2,.3,.4,.5]):
            d=frame[(frame.kinematic_uncertainty_distance>=lo)&(frame.kinematic_uncertainty_distance<hi if hi<.5 else frame.kinematic_uncertainty_distance<=hi)]
            bins.append(dict(scenario=scenario,distance_low=lo,distance_high=hi,n=len(d),
                corrected=int(d.outcome.eq('corrected').sum()),harmed=int(d.outcome.eq('harmed').sum()),
                mean_abs_delta_cwt=float(d.delta_cwt.abs().mean()) if len(d) else None))
    pd.DataFrame(bins).to_csv(OUT/'fusion_confidence_bins.csv',index=False)
    cwt=pd.concat([pd.read_csv(OUT/'folds'/f'cwt_components_{f}.csv') for f in range(6)],ignore_index=True)
    cwt.to_csv(OUT/'cwt_component_importance.csv',index=False)
    adaptive=fusion[fusion.scenario=='adaptive__KC']; cases=[]
    for status in ('corrected','harmed','unchanged_correct','unchanged_wrong'):
        d=adaptive[adaptive.outcome==status].sort_values('participant_id')
        if len(d): cases.append(dict(outcome=status,participant_id=d.participant_id.iloc[0],fold=int(d.fold.iloc[0]),
                                    selection_rule='first lexicographic participant in category; post-hoc illustrative only'))
    write_json(OUT/'local_cases.json',cases)
    original=json.loads((OUT/'original_artifacts.json').read_text())
    for p,h in original.items():
        if sha(ROOT/p)!=h: raise ValueError('Old artifact modified: '+p)
    write_report(tables,stable,per_summary,pd.DataFrame(concordance),outcome_summary,fold_summary,fusion,len(original))
    numeric={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file()
        and p.suffix in ('.json','.csv','.npz','.md') and p.name not in ('validation.json','notebook_validation.json','test_validation.json')
        and not any(p.is_relative_to(OUT/d) for d in ('figures','attempts'))}
    write_json(OUT/'validation.json',dict(complete=False,stage='numerical_validation_passed',numerical_validation_passed=True,
        n_subjects=258,outer_folds=6,seed=42,n_features=1036,n_tasks=14,n_source_groups=12,
        outer_test_attribution_rows=258,fusion_rows=516,permutation_blocks=26,permutation_repeats_per_block_fold=30,
        permutation_evaluations=len(perm),baseline_oof_matches=True,fusion_oof_matches=True,
        shap_additivity_passed=True,cwt_linear_additivity_passed=True,all_permutations_replayed=True,
        mapping_source_sentinels_passed=True,train_only_backgrounds_verified=True,original_artifacts_unchanged=len(original),
        numeric_artifact_sha256=numeric))
    print('XAI NUMERICAL VALIDATION PASSED',flush=True)


def write_report(tables,stable,permutation,concordance,outcome_summary,mechanisms,fusion,n_old):
    features=tables['feature'].head(20)[['task','feature','group','sum_mean_abs_shap','top_k_fold_count']]
    groups=tables['group'][['name','n_features','sum_mean_abs_shap','per_feature_mean_abs','attribution_mass_share_percent']]
    tasks=tables['task'][['name','sum_mean_abs_shap','attribution_mass_share_percent','mean_rank','sd_rank','top_k_fold_count']]
    perm=permutation[permutation.level=='task'][['block','mean_drop_f1_macro_pp','fold_sd_drop_f1_macro_pp','mean_drop_auc_pp']].sort_values('mean_drop_f1_macro_pp',ascending=False)
    counts=outcome_summary.pivot(index='scenario',columns='outcome',values='n').fillna(0).reset_index()
    first=', '.join(tables['task'].name.head(3)); topgroups=', '.join(tables['group'].name.head(3))
    pct=float(tables['task'].attribution_mass_share_percent.head(3).sum())
    adaptive=fusion[fusion.scenario=='adaptive__KC']; active=adaptive.gate_status.eq('active')
    sensitivity=pd.read_csv(OUT/'explainer_sensitivity.csv')
    report=f'''# Explainable AI — Kinematic và Fusion Kinematic+CWT

## Kết quả chính và phạm vi

Giải thích **mô hình đã đánh giá**, không thay kiến trúc, classifier, reducer, trọng số hoặc threshold.
Giữ **258 participant, nhãn group, sáu outer folds và seed 42**. Attribution chỉ tính trên 43 outer Test
của mô hình fit trên 215 outer Train tương ứng. Baseline OOF và hai Fusion được replay khớp trước giải thích.
Trạng thái cuối cùng chỉ lấy từ `validation.json complete=true`, sau tests, notebook và hình.

Theo attribution mass của raw TreeSHAP OOF, ba tác vụ đứng đầu là **{first}**,
chiếm **{pct:.2f}%** tổng mean absolute attribution. Ba nhóm có tổng mass cao nhất là **{topgroups}**.
Đây là mô tả việc mô hình sử dụng thông tin, không phải bằng chứng nhân quả, biomarker hay chẩn đoán lâm sàng.

## Mapping thực tế và preprocessing

`feature_mapping.csv` ánh xạ đủ 1.036 cột = 14 tác vụ × 74 đặc trưng. Thứ tự task lấy từ
`data/derived/schema.json` và kiểm tra bằng `prepare.py TASKS`; tên/thứ tự feature kiểm tra trực tiếp
từ `features.py kinematic()` trên raw source. Tái dựng đủ 14 task của participant đầu, giữa, cuối
(chọn không dùng nhãn) và đối chiếu toàn bộ 1.036 giá trị cache; xem `mapping_audit.json`.
Không giả định tên cột từ vị trí hoặc tự đổi thứ tự tác vụ. Metadata, cognitive scores và ID không là feature.

Nhóm lấy theo **12 prefix thực tế trong code**: Velocity, Acceleration, Jerk, Peaks, Lifts,
Pressure, Orientation, Trajectory, Curvature, Statistics, Energy, FFT. Statistics giữ các entropy/
skewness/kurtosis của nhiều tín hiệu; Energy giữ RMS; không đếm lặp các feature này vào Pressure/Velocity.
Đây là taxonomy của implementation, không ép thành số nhóm được mô tả trong bài báo gốc.

SHAP chạy trên đầu vào sau median imputer → bỏ cột hằng → StandardScaler đã fit trong outer Train.
Ánh xạ qua support của VarianceThreshold để trả về đúng 1.036 tọa độ; cột bị bỏ có SHAP=0.
Không inverse-transform SHAP bằng scaler: attribution vẫn ở output space; giá trị X trong hình là
giá trị raw để đọc đơn vị đầu vào. Scaler affine theo từng cột không làm đổi danh tính feature.

## SHAP output space và kiểm tra cộng dồn

SHAP **0.49.1**, tương thích Python 3.10 của môi trường hiện có; NumPy/sklearn/LightGBM cũ không bị nâng.
Tính bằng `TreeExplainer(...)(X_test)` và API `Explanation`, `approximate=False`.

- **Primary:** `tree_path_dependent`, `model_output=raw`. Reference là training cover/leaf counts
  của chính model outer Train, không lấy Test làm background.
- **Sensitivity:** `interventional raw` và `interventional probability`, cùng 100 background rows
  lấy ngẫu nhiên seed 42 **chỉ từ outer Train**, không stratify hoặc chọn theo nhãn/SHAP.
- Kiểm tra cả 258 người: `base_raw + Σφ_raw = predict(raw_score=True)`;
  với LightGBM binary này, đã xác minh raw score = `log(p/(1−p))` và sigmoid(raw)=p(group=1).
- Với probability SHAP: `base_probability + Σφ_probability = predict_proba[:,1]`.
  Probability SHAP có thể âm; dấu dương đẩy về group=1, không đồng nghĩa "gây bệnh".
- **Không lấy sigmoid từng φ_raw rồi cộng** để thành probability attribution. Kể cả cùng background,
  `E[sigmoid(raw)]` thường khác `sigmoid(E[raw])`; bảng `output_space_checks.csv` giữ hai reference
  để kiểm tra. Primary raw/path và probability/interventional còn khác giả định masking nên không
  diễn giải khác biệt giữa chúng như chỉ do hàm sigmoid.

**Kiểm chứng backend:** SHAP 0.49 independent engine lưu threshold trong C++ `Node.thres` ở float32.
Lượt thử `attempts/precision_02/` phát hiện residual raw ~6.46e−4 và probability ~7.78e−5 ở fold 2,
dù kiểm tra mặc định của SHAP vẫn qua. Không tăng tolerance hoặc cộng offset để ép khớp.
Thay vào đó chỉ dùng cho **explainer copy** một lossless partition encoding: với mỗi feature, lấy
các split thresholds float64 từ **frozen model**, ánh xạ input vào bin theo `searchsorted(side=left)`;
thay threshold thứ j bởi j+0.5. Mọi phép so sánh x≤t tương đương bin(x)≤j+0.5, kể cả đúng tại t.
Giữ nguyên leaf values, cây, feature players và missing routing; checkpoint/pipeline gốc không đổi.
Do phép biến đổi theo từng cột bảo toàn mọi split, mọi feature-masked coalition có cùng prediction,
nên game interventional không đổi. Cuts chỉ đến từ model đã fit Train, không học từ Test hay nhãn.
Unit test kiểm tra threshold sát ranh giới và SHAP của stump bằng tính trực tiếp; mỗi fold kiểm tra
toàn bộ split boundaries, Train background, Test và 300 hybrid masks đều khớp raw model tới 1e−12.
Attribution sau đó vẫn phải qua tolerance 2e−6 và kiểm tra probability/raw độc lập.

Attribution chịu ảnh hưởng của tương quan feature và lựa chọn reference/masking; interventional không
biến dữ liệu quan sát thành bằng chứng nhân quả. Kết quả độ nhạy thứ hạng:

{mdtable(sensitivity)}

## Top Features — thang log-odds

`sum_mean_abs_shap` ở hàng feature là mean |φ| trên 258 outer Test rows; `top_k_fold_count`
là số fold feature thuộc Top-20. Cột này chỉ mô tả ổn định, không dùng chọn lại feature/model.

{mdtable(features)}

## Top Feature Groups

{mdtable(groups)}

Tổng mass có xu hướng lớn hơn khi nhóm có nhiều cột. Vì vậy giữ thêm `per_feature_mean_abs` và
số cột; không suy rằng FFT quan trọng sinh học hơn chỉ vì có nhiều tọa độ.
Hai định nghĩa được lưu riêng: **mass = E[Σ|φ_j|]** và **net magnitude = E[|Σφ_j|]**.
Signed sum theo group/task vẫn cộng đúng về output − base; mass không có tính cộng dồn về prediction.
Đây là **aggregation của individual SHAP**, không phải Group SHAP/Owen values tính bằng joint masker.

## Top Tasks và độ ổn định sáu fold

{mdtable(tasks)}

Thứ hạng theo tổng mass; mỗi tác vụ có cùng 74 feature nên tránh khác biệt số cột giữa task.
`mean_rank ± sd_rank`, Top-3 frequency và heatmap sáu fold phản ánh cả thay đổi model lẫn 43 Test rows.
Train folds chồng lấp; SD không phải CI, 15 cặp fold không phải 15 mẫu độc lập.

{mdtable(stable)}

## Grouped permutation trên held-out Test

Hoán vị đồng thời toàn bộ 74 cột của một task bằng cùng một permutation người; làm thêm 12 block
feature group xuyên các task. **30 lần/block/fold, 26 block × 30 × 6 = 4.680** đánh giá prediction.
Giữ model/preprocessing cố định, không fit lại và không chọn threshold. Chỉ dùng Test labels để tính
chênh F1/Accuracy/AUC/Balanced Accuracy. Mọi permutation và prediction được lưu, replay lại trong validator.

Đơn vị **điểm phần trăm**: điểm gốc trừ điểm sau hoán vị. Bảng là mean sáu fold (mỗi fold đã lấy mean
30 lần); SD là giữa sáu fold, không gộp 180 lần như mẫu độc lập. Giữ cả giá trị âm.

{mdtable(perm)}

Đối chiếu thứ hạng SHAP mass và permutation F1:

{mdtable(concordance)}

Đây là marginal/joint-block permutation: bảo toàn cấu trúc trong block nhưng phá quan hệ với các
block còn lại, có thể tạo tổ hợp ngoài phân phối. Tương quan/redundancy và chỉ 43 người/fold có thể
làm SHAP và permutation khác nhau; không chọn chỉ metric nào thuận lợi hoặc coi permutation là nhân quả.

**Chưa thể kết luận số tác vụ tối thiểu.** Các tác vụ mass thấp vẫn có thể bổ sung cho nhau;
không cộng Top-k mass thành accuracy giữ lại. Đề xuất một task ablation riêng: chọn subset hoặc
xếp hạng task trong inner Train, refit toàn pipeline trên subset trong mỗi split, đối chiếu cả
accuracy/F1/AUC và thời gian làm bài trên outer Test, xác nhận bằng cohort ngoài. Không chạy thí nghiệm
giảm task hoặc tối ưu accuracy trong nhiệm vụ XAI này.

## Fusion: xác suất, trọng số và confidence gate

`fusion_participants.csv` có đủ **516 hàng** (258 × Late/Adaptive): pK, pC, pFusion, native predictions,
effective weights, nominal weights, gate status/threshold/budget, Δp do CWT và outcome.
Tất cả probability được tái tạo từ đúng branch checkpoint/combiner rồi đối chiếu `nested_oof.csv`.

- **Late:** pFusion = wK pK + wC pC, wK+wC=1. Không có confidence gate;
  gate status ghi `not_applicable`. Δp_CWT = wC(pC−pK).
- **Adaptive:** g=1[|pK−0.5|≤τ]; pFusion=pK+gα(pC−pK). Có một nhánh phụ nên auxiliary simplex
  weight bằng 1; α và τ đã được chọn trong inner CV từ nghiên cứu trước. Effective wC=gα,
  effective wK=1−gα. Đây là **confidence-gated probability correction**, **không phải learned attention network**.
  Khi g=0, pFusion=pK đúng số học. {int(active.sum())}/{len(adaptive)} participant có gate active.
- Δp là thay đổi xác suất **so với Kinematic**, không phải SHAP hay causal contribution của một modality.
  Không nhân raw Kinematic SHAP với một weight rồi gọi là probability SHAP của Fusion. Gate phụ thuộc
  pK tạo tương tác/điểm chuyển trạng thái; giải thích ở đây là replay cơ chế thực tế.

{mdtable(counts)}

Late sửa đúng 9 người và làm sai 8 người (ròng +1); Adaptive sửa đúng 8, làm sai 6 (ròng +2).
Các con số khớp nghiên cứu Fusion trước, không phải kết quả mô hình mới.
Trên toàn cohort, pC không tự giúp nếu quá yếu hoặc kéo một pK đúng qua ngưỡng 0.5.
`fusion_confidence_bins.csv` và scatter Δp cho thấy khi nào correction thay nhãn hoặc chỉ thay độ tự tin.
Đây là phân tích hậu nghiệm, không chỉnh gate theo các ca Test vừa quan sát.

Tham số/nhánh CWT thực tế từng fold:

{mdtable(mechanisms)}

Baseline Kinematic Accuracy OOF 84.88%, F1 84.72%; Late KC 85.27%/85.10%; Adaptive KC 85.66%/85.56%.
Mức tăng nhỏ và KC là family đứng đầu sau xem outer results; `all_selected` trước đó chỉ đạt
81.78%/81.52%. XAI không làm tăng bằng chứng tổng quát hóa, không dùng để thay baseline tự động.

## CWT/PCA: giới hạn diễn giải

CWT ở hai pipeline dùng **LogisticRegression**, không dùng TreeExplainer cho model này.
Lưu exact independent-reference linear decomposition: φ_j = β_j(z_j−mean_background_train(z_j));
base = intercept + β·mean_background, cộng về **CWT log-odds**; sigmoid tổng khớp pC.
Phép trừ/mean attribution tính float64 để không mất đẳng thức với input cache float32.
Lượt kiểm tra ban đầu phát hiện lỗi cỡ 8.5e−8 do float32 subtraction; giữ ở
`attempts/precision_01/` với trạng thái chưa hoàn tất, không nới tolerance để che lỗi.

PCA attribution chỉ có nghĩa ở **tọa độ embedding/PCA của model đó**. `cwt_component_importance.csv`
giữ fold/scenario/reducer và component index riêng; **không gộp PC1 giữa các fold**, vì PCA axes/signs
được fit riêng. Với `none`, tên là preprocessed embedding coordinate, không phải đặc trưng sinh học.
Không back-project attribution qua encoder, không gán PC cho task/tín hiệu, không diễn giải thành
thang tần số CWT hoặc ý nghĩa sinh học khi chưa có mapping attribution hợp lệ.

## Hình/bảng đáng giữ để viết paper

- **Main:** SHAP beeswarm/bar (raw log-odds), task importance với fold SD, task rank heatmap,
  grouped permutation tasks và Fusion corrected/harmed + Δp scatter.
- **Supplement:** Summary plot (cùng dữ liệu với beeswarm, không phải bằng chứng độc lập),
  dependence scatter, bốn waterfall cho corrected/harmed/unchanged correct/unchanged wrong,
  group mass so với per-feature average, task×group heatmap, task-level waterfall và CWT PCA bar.
- Hình dependence là quan hệ X–SHAP quan sát qua sáu mô hình, không phải partial dependence,
  can thiệp nhân quả hay khuyến nghị thay đổi hành vi. Waterfall lấy participant đầu theo ID của
  từng outcome, chỉ để minh họa; `local_cases.json` ghi rule và fold, không chọn ca có hình đẹp nhất.

Không kết luận marker bệnh, tối thiểu bao nhiêu task hay tính an toàn lâm sàng từ cohort này.
Giải thích mô hình phụ thuộc bộ feature dựng lại và nhãn `group` (khác diagnosis flag cụ thể).
Cần ablation có thiết kế và external validation riêng cho kết luận về task reduction/tổng quát hóa.

## Artifacts và chạy lại

{n_old} artifact cũ được kiểm tra SHA-256, không bị sửa. Tất cả kết quả mới ở `results/xai/`;
notebook mới `notebooks/09_xai.ipynb`, code `scripts/*xai*.py`, module `diagramo/explain.py`, tests riêng.
`attributions/` giữ ma trận SHAP 258×1036 và linear CWT contributions; `folds/` giữ participant IDs,
background IDs, errors, SHA và cơ chế; `permutation/` giữ mọi permutation/prediction; CSV cấp feature,
task/group/fold/outcome phục vụ viết paper. Không có attribution từ model train full cohort.

```powershell
.venv/Scripts/python.exe -m pip install --no-deps -r requirements-xai.txt
.venv/Scripts/python.exe -X utf8 -B -u scripts/analyze_xai.py
.venv/Scripts/python.exe -X utf8 -B -u scripts/report_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/plot_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_xai_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 09_xai.ipynb --validation-output results/xai/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_xai.py
```

Tham khảo chính thức: [TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html),
[LinearExplainer](https://shap.readthedocs.io/en/stable/generated/shap.LinearExplainer.html),
[Permutation importance](https://scikit-learn.org/1.7/modules/permutation_importance.html).
'''
    (OUT/'REPORT_VI.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
