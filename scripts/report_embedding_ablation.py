"""Audit all folds/OOF and write Vietnamese embedding ablation conclusions."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from embedding_common import ROOT,OUT,BRANCHES,MODES,CLASSIFIERS,REDUCERS,PRIMARY,sha,write_json
from evaluate import metrics
from evaluate_embedding_ablation import load_subjects,select_from_inner


def table(df):
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def aggregate(d,scores):
    result=metrics(d.label,d.prediction,d.probability)
    for m in list(result):
        result['mean_'+m]=float(np.mean([r[m] for r in scores]))
        result['std_'+m]=float(np.std([r[m] for r in scores],ddof=1))
    return result


def validate_record(r,ids,y,folds,signature):
    f=r['fold']; tr=np.flatnonzero(folds.fold.to_numpy()!=f); te=np.flatnonzero(folds.fold.to_numpy()==f)
    train=ids[tr].tolist(); test=ids[te].tolist()
    if r['evaluation_sha256']!=signature: raise ValueError('Fold settings changed')
    if r['train_participants']!=train or r['test_participants']!=test or r['test_indices']!=te.tolist():
        raise ValueError('Outer participant identity mismatch')
    if r['bank']['fit_participants']!=train or set(train)&set(test): raise ValueError('Outer transform leakage')
    inner_splits=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=42).split(ids[tr],y[tr]))
    if len(r['inner_audits'])!=3: raise ValueError('Missing inner audit')
    for audit,(itr,iva) in zip(r['inner_audits'],inner_splits):
        fit_ids=ids[tr][itr].tolist(); val_ids=ids[tr][iva].tolist()
        if (audit['train_participants']!=fit_ids or audit['validation_participants']!=val_ids
                or audit['bank']['fit_participants']!=fit_ids or set(fit_ids)&set(val_ids)
                or (set(fit_ids)|set(val_ids))&set(test)):
            raise ValueError('Inner transform leakage or split mismatch')
    expected_reducers=(list(r['reduction_metadata']) if r['mode']=='original' else [s['name'] for s in REDUCERS])
    expected={(reducer,c) for reducer in expected_reducers for c in CLASSIFIERS}
    if len(r['scores'])!=len(expected) or {(s['reducer'],s['classifier']) for s in r['scores']}!=expected:
        raise ValueError('Outer candidate coverage mismatch')
    inner=pd.DataFrame(r['inner_scores'])
    if len(inner)!=3*len(expected): raise ValueError('Inner grid incomplete')
    for k in range(3):
        frame=inner[inner.inner_fold==k]
        if {(s.reducer,s.classifier) for s in frame.itertuples()}!=expected:
            raise ValueError('Inner candidate identity mismatch')
    for score in r['scores']:
        metadata=r['reduction_metadata'][score['reducer']]
        if score['n_features']!=metadata['n_features']: raise ValueError('Feature width mismatch')
        if metadata['method'] in ('anova','extratrees'):
            indices=metadata['selected_original_indices']
            if (len(indices)!=score['n_features'] or len(set(indices))!=len(indices)
                    or min(indices)<0 or max(indices)>=r['bank']['raw_features']):
                raise ValueError('Invalid selected feature indices')
        p=r['predictions'][score['reducer']+'__'+score['classifier']]
        pred=np.array(p['prediction']); prob=np.array(p['probability'])
        if (pred.shape!=(43,) or prob.shape!=(43,) or not np.isin(pred,[0,1]).all()
                or not np.isfinite(prob).all() or not ((prob>=0)&(prob<=1)).all()):
            raise ValueError('Invalid held-out predictions')
        for m,v in metrics(y[te],pred,prob).items():
            if not np.isclose(v,score[m],atol=1e-12,rtol=0): raise ValueError('Metric differs from saved predictions')
    if r['mode']=='original':
        if sha(ROOT/r['source'])!=r['source_sha256']: raise ValueError('Prior result changed')
        old=json.loads((ROOT/r['source']).read_text(encoding='utf-8'))
        if r['predictions']!=old['predictions'] or r['inner_scores']!=old['selection']['inner_fold_scores']:
            raise ValueError('Original OOF/inner scores were altered')


def generate_report():
    write_json(OUT/'validation.json',dict(complete=False,stage='validating'))
    ids,y,folds,_=load_subjects()
    pd.testing.assert_frame_equal(pd.read_csv(OUT/'fold_assignments.csv'),folds)
    signature=sha(OUT/'evaluation_settings.json')
    config=json.loads((OUT/'settings.json').read_text(encoding='utf-8'))
    eval_config=json.loads((OUT/'evaluation_settings.json').read_text(encoding='utf-8'))
    if eval_config['extraction_settings_sha256']!=sha(OUT/'settings.json'): raise ValueError('Settings mismatch')
    for mapping in (config['input_sha256'],config['code_sha256'],eval_config['code_sha256']):
        for p,h in mapping.items():
            if sha(ROOT/p)!=h: raise ValueError('Input/source changed: '+p)
    for p,h in config['checkpoint_sha256'].items():
        if sha(ROOT/'.cache/torch/hub/checkpoints'/p)!=h: raise ValueError('Original checkpoint changed')
    old=json.loads((OUT/'original_artifacts.json').read_text(encoding='utf-8'))
    for p,h in old.items():
        if sha(ROOT/p)!=h: raise ValueError('Old artifact changed: '+p)
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'folds').glob('*.json'))]
    expected={(b,m,f) for b in BRANCHES for m in MODES for f in range(6)}
    if len(records)!=48 or {(r['branch'],r['mode'],r['fold']) for r in records}!=expected:
        raise ValueError('All 48 outer records must finish before validation')
    for r in records: validate_record(r,ids,y,folds,signature)
    for branch in BRANCHES:
        meta=json.loads((OUT/'embeddings'/f'{branch}.json').read_text())
        if meta['settings_sha256']!=sha(OUT/'settings.json') or meta['label_access'] or not meta['frozen']:
            raise ValueError('Embedding extraction provenance invalid')
        for item in meta['outputs'].values():
            path=OUT/'embeddings'/item['file']
            if sha(path)!=item['sha256']: raise ValueError('Embedding matrix hash differs')
            a=np.load(path,mmap_mode='r')
            if list(a.shape)!=item['shape'] or a.shape[0]!=258 or not np.isfinite(a).all():
                raise ValueError('Invalid embedding matrix')
    oof_dir=OUT/'oof'; oof_dir.mkdir(exist_ok=True)
    rows=[]; fold_rows=[]; cost_rows=[]; job_costs=[]; selected=[]; nested_rows=[]; nested_oof=[]
    def collate(six,reducer,clf):
        d=folds.copy(); d['prediction']=0; d['probability']=np.nan; scores=[]
        for r in six:
            scores.append(next(s for s in r['scores'] if s['reducer']==reducer and s['classifier']==clf))
            p=r['predictions'][reducer+'__'+clf]; ix=r['test_indices']
            d.loc[ix,'prediction']=p['prediction']; d.loc[ix,'probability']=p['probability']
        if d.probability.isna().any(): raise ValueError('OOF coverage gap')
        return d,scores
    lookup={(r['branch'],r['mode'],r['fold']):r for r in records}
    for branch in BRANCHES:
        for mode in MODES:
            six=[lookup[branch,mode,f] for f in range(6)]
            for r in six:
                job_costs.append(dict(branch=branch,mode=mode,outer_fold=r['fold'],
                                      seconds=r.get('seconds',np.nan),reused_original=r['reused_original']))
                for split,audit in [('outer',r),*[('inner_'+str(a['inner_fold']),a) for a in r['inner_audits']]]:
                    cost_rows.append(dict(branch=branch,mode=mode,outer_fold=r['fold'],split=split,
                        bank_fit_seconds=audit.get('bank_fit_seconds',np.nan),
                        bank_transform_seconds=audit.get('bank_transform_seconds',np.nan),
                        reused_original=r['reused_original']))
            for spec in REDUCERS:
                for clf in CLASSIFIERS:
                    d,scores=collate(six,spec['name'],clf)
                    if mode=='original':
                        reference=pd.read_csv(ROOT/'results/feature_optimization/oof'/f'{branch}__{spec["name"]}__{clf}.oof.csv',float_precision='round_trip')
                        np.testing.assert_array_equal(d.prediction,reference.prediction)
                        np.testing.assert_array_equal(d.probability,reference.probability)
                    d.to_csv(oof_dir/f'{branch}__{mode}__{spec["name"]}__{clf}.oof.csv',index=False)
                    for r,score in zip(six,scores):
                        fold_rows.append(dict(branch=branch,mode=mode,fold=r['fold'],**score))
                    rows.append(dict(branch=branch,mode=mode,reducer=spec['name'],method=spec['method'],
                        classifier=clf,raw_features=six[0]['bank']['raw_features'],
                        mean_features=float(np.mean([s['n_features'] for s in scores])),
                        min_features=min(s['n_features'] for s in scores),max_features=max(s['n_features'] for s in scores),
                        mean_classifier_fit_seconds=(np.nan if mode=='original' else float(np.mean([s['classifier_fit_seconds'] for s in scores]))),
                        **aggregate(d,scores)))
        branch_selections=[]
        for f in range(6):
            inner=[dict(mode=mode,**score) for mode in MODES for score in lookup[branch,mode,f]['inner_scores']]
            selections,ranking=select_from_inner(inner)
            branch_selections.append(selections)
            write_json(OUT/'selections'/f'{branch}__fold_{f}.json',dict(branch=branch,fold=f,
                train_participants=lookup[branch,'original',f]['train_participants'],
                inner_candidates=ranking,selections=selections,selection_input='inner fold scores only'))
        for scenario in branch_selections[0]:
            d=folds.copy(); d['prediction']=0; d['probability']=np.nan; scores=[]; dims=[]
            for f,selections in enumerate(branch_selections):
                choice=selections[scenario]; r=lookup[branch,choice['mode'],f]
                p=r['predictions'][choice['reducer']+'__'+choice['classifier']]
                score=next(s for s in r['scores'] if s['reducer']==choice['reducer'] and s['classifier']==choice['classifier'])
                ix=r['test_indices']; d.loc[ix,'prediction']=p['prediction']; d.loc[ix,'probability']=p['probability']
                scores.append(score); dims.append(score['n_features'])
                selected.append(dict(branch=branch,scenario=scenario,fold=f,**choice,n_features=score['n_features'],
                                     **{'outer_'+m:score[m] for m in PRIMARY}))
            if d.probability.isna().any(): raise ValueError('Nested OOF coverage gap')
            if scenario in ('original_raw','original_compact'):
                previous=pd.read_csv(ROOT/'results/feature_optimization/nested_oof.csv',float_precision='round_trip')
                previous=previous[(previous.branch==branch)&(previous.scenario==('baseline' if scenario=='original_raw' else 'optimized'))]
                np.testing.assert_array_equal(d.prediction,previous.prediction)
                np.testing.assert_array_equal(d.probability,previous.probability)
            nested_oof.append(d.assign(branch=branch,scenario=scenario))
            nested_rows.append(dict(branch=branch,scenario=scenario,mean_features=float(np.mean(dims)),**aggregate(d,scores)))
    summary=pd.DataFrame(rows).sort_values(['branch','mean_f1_macro','mean_accuracy','mean_auc'],ascending=[True,False,False,False])
    if len(summary)!=192: raise ValueError('Expected 192 shared-grid candidates')
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'fold_scores.csv',index=False)
    pd.DataFrame(cost_rows).to_csv(OUT/'fit_costs.csv',index=False)
    pd.DataFrame(job_costs).to_csv(OUT/'cv_job_costs.csv',index=False)
    selected=pd.DataFrame(selected); selected.to_csv(OUT/'selected_by_fold.csv',index=False)
    pd.concat(nested_oof,ignore_index=True).to_csv(OUT/'nested_oof.csv',index=False)
    nested=pd.DataFrame(nested_rows); nested.to_csv(OUT/'nested_summary.csv',index=False)
    comparisons=summary.copy()
    original=summary[summary['mode']=='original'].set_index(['branch','reducer','classifier'])
    raw=summary[summary.reducer=='none'].set_index(['branch','mode','classifier'])
    for m in PRIMARY:
        comparisons['delta_adaptation_'+m+'_pp']=[100*(r['mean_'+m]-original.loc[(r.branch,r.reducer,r.classifier),'mean_'+m]) for _,r in summary.iterrows()]
        comparisons['delta_reduction_'+m+'_pp']=[100*(r['mean_'+m]-raw.loc[(r.branch,r['mode'],r.classifier),'mean_'+m]) for _,r in summary.iterrows()]
    comparisons.to_csv(OUT/'ablation_matched.csv',index=False)
    nested_comparison=[]
    for branch in BRANCHES:
        for _,r in nested[nested.branch==branch].iterrows():
            row=dict(branch=branch,scenario=r.scenario,mean_features=r.mean_features)
            for comparator in ('original_raw','original_compact'):
                base=nested[(nested.branch==branch)&(nested.scenario==comparator)].iloc[0]
                for m in PRIMARY: row['delta_vs_'+comparator+'_'+m+'_pp']=100*(r['mean_'+m]-base['mean_'+m])
            nested_comparison.append(row)
    nested_comparison=pd.DataFrame(nested_comparison); nested_comparison.to_csv(OUT/'nested_comparison.csv',index=False)
    prior=pd.read_csv(ROOT/'results/feature_optimization/best_per_branch.csv')
    prior[prior.branch.isin(BRANCHES)].to_csv(OUT/'previous_grid_winners.csv',index=False)
    benchmark=json.loads((OUT/'benchmark.json').read_text())
    if benchmark['settings_sha256']!=sha(OUT/'settings.json') or len(benchmark['records'])!=8:
        raise ValueError('Matched benchmark missing')
    baseline_audit=json.loads((OUT/'baseline_extraction_audit.json').read_text())
    if (baseline_audit['settings_sha256']!=sha(OUT/'settings.json') or not baseline_audit['all_tasks_verified']
            or len(baseline_audit['records'])!=2 or not all(r['tasks_checked']==3612 and r['allclose_passed'] for r in baseline_audit['records'])):
        raise ValueError('Full-cohort baseline extraction equivalence not verified')
    pd.DataFrame(benchmark['records']).drop(columns=['seconds']).to_csv(OUT/'extraction_costs.csv',index=False)
    write_report(summary,nested,nested_comparison,selected,benchmark)
    numeric_files=[p for folder in ('folds','oof','selections') for p in (OUT/folder).glob('*') if p.is_file()]
    numeric_files += [p for p in OUT.glob('*.csv')]
    numeric_files += [OUT/p for p in ('settings.json','evaluation_settings.json','benchmark.json',
                                      'baseline_extraction_audit.json','REPORT_VI.md')]
    write_json(OUT/'validation.json',dict(complete=False,numerical_validation_passed=True,
        stage='numerics validated; notebook and figures pending',fixed_results=192,outer_records=48,
        new_outer_records=36,new_inner_fit_partitions=108,original_artifacts_unchanged=len(old),
        original_fixed_and_nested_oof_exact=True,participant_identity_and_folds_verified=True,
        all_metrics_recomputed_from_oof=True,train_only_transform_audits_passed=True,
        encoder_and_adapter_frozen=True,seed=42,n_subjects=258,
        numerical_artifact_sha256={str(p.relative_to(OUT)):sha(p) for p in numeric_files}))
    print('NUMERICS VALIDATED: 192 fixed candidates, 20 nested scenarios, 48 outer records',flush=True)


def write_report(summary,nested,comparison,selected,benchmark):
    def triplet(r): return ' / '.join(f'{100*r["mean_"+m]:.2f}' for m in PRIMARY)
    fixed=[]
    for branch in BRANCHES:
        for mode in MODES:
            for compact in (False,True):
                g=summary[(summary.branch==branch)&(summary['mode']==mode)]
                g=g[g.reducer!='none'] if compact else g[g.reducer=='none']
                r=g.iloc[0]
                fixed.append(dict(Nhánh=branch,Embedding=mode,Giảm_chiều='có' if compact else 'không',
                    Cấu_hình=r.reducer+' / '+r.classifier,Chiều=r.mean_features,Acc_F1_AUC=triplet(r)))
    nested_table=[]; decisions=[]
    for branch in BRANCHES:
        for _,r in nested[nested.branch==branch].iterrows():
            change=comparison[(comparison.branch==branch)&(comparison.scenario==r.scenario)].iloc[0]
            nested_table.append(dict(Nhánh=branch,Quy_trình=r.scenario,Chiều_TB=r.mean_features,
                Acc_F1_AUC=triplet(r),Δ_F1_vs_gốc=change.delta_vs_original_raw_f1_macro_pp,
                Δ_F1_vs_giảm_chiều_cũ=change.delta_vs_original_compact_f1_macro_pp))
        candidate=nested[(nested.branch==branch)&(nested.scenario=='new_selected')].iloc[0]
        change=comparison[(comparison.branch==branch)&(comparison.scenario=='new_selected')].iloc[0]
        verdict=('có cải thiện mô tả trên cùng các outer fold' if change.delta_vs_original_raw_f1_macro_pp>0
                 else 'chưa cải thiện F1 so với quy trình gốc')
        decisions.append(f'- **{branch}:** quy trình chọn trong các embedding mới {verdict}: '
            f'F1 {100*candidate.mean_f1_macro:.2f}%, Δ so với gốc {change.delta_vs_original_raw_f1_macro_pp:+.2f} điểm, '
            f'Δ so với quy trình giảm chiều cũ {change.delta_vs_original_compact_f1_macro_pp:+.2f} điểm. '
            'Đây là kết quả chọn toàn quy trình, không chứng minh riêng một cấu hình cố định là tối ưu.')
    frequency=selected.groupby(['branch','scenario','mode','reducer','classifier']).size().reset_index(name='outer_folds')
    frequency.to_csv(OUT/'selection_frequency.csv',index=False)
    candidates=[]
    for branch in BRANCHES:
        votes=selected[(selected.branch==branch)&(selected.scenario=='new_selected')]['mode'].value_counts()
        for mode in MODES[1:]:
            score=nested[(nested.branch==branch)&(nested.scenario==mode+'_compact')].iloc[0]
            change=comparison[(comparison.branch==branch)&(comparison.scenario==mode+'_compact')].iloc[0]
            cost=next(r for r in benchmark['records'] if r['branch']==branch and r['mode']==mode)
            base_cost=next(r for r in benchmark['records'] if r['branch']==branch and r['mode']=='original')
            candidates.append(dict(branch=branch,mode=mode,inner_selected_outer_folds=int(votes.get(mode,0)),
                nested_mean_f1_macro=score.mean_f1_macro,
                nested_mean_features=score.mean_features,raw_features=cost['raw_features'],
                inference_time_ratio_vs_original=cost['median_seconds']/base_cost['median_seconds'],
                delta_vs_original_raw_f1_macro_pp=change.delta_vs_original_raw_f1_macro_pp,
                delta_vs_original_compact_f1_macro_pp=change.delta_vs_original_compact_f1_macro_pp,
                recommendation=('candidate_for_future_fusion_ablation'
                    if change.delta_vs_original_raw_f1_macro_pp>0 else 'retain_as_control_only')))
    candidates=pd.DataFrame(candidates)
    candidates.to_csv(OUT/'fusion_candidates.csv',index=False)
    candidate_table=candidates.copy()
    candidate_table['nested_F1_percent']=100*candidate_table.pop('nested_mean_f1_macro')
    interpretation=[]
    for branch in BRANCHES:
        def scenario(name): return nested[(nested.branch==branch)&(nested.scenario==name)].iloc[0]
        base=scenario('original_raw'); raw=scenario('channel_concat_raw'); compact=scenario('channel_concat_compact')
        raw_delta=100*(raw.mean_f1_macro-base.mean_f1_macro)
        compact_delta=100*(compact.mean_f1_macro-raw.mean_f1_macro)
        auc_delta=100*(compact.mean_auc-raw.mean_auc)
        interpretation.append(f'- **{branch}, tách adaptation khỏi nén:** `channel_concat_raw` có F1 '
            f'{100*raw.mean_f1_macro:.2f}% so với {100*base.mean_f1_macro:.2f}% của `original_raw` '
            f'({raw_delta:+.2f} điểm). Khi thêm tìm kiếm reducer, F1 thành {100*compact.mean_f1_macro:.2f}% '
            f'({compact_delta:+.2f} điểm so với concat raw), AUC thay đổi {auc_delta:+.2f} điểm; '
            f'số chiều từ {raw.mean_features:.0f} xuống trung bình {compact.mean_features:.2f}. '
            'Đây là đối chiếu hai quy trình nested; classifier có thể được chọn khác nhau. '
            'Để cô lập từng phép biến đổi với classifier cố định, xem ablation matched.')
    costs=pd.DataFrame(benchmark['records'])[['branch','mode','raw_features','median_seconds',
                                               'estimated_ms_per_participant','peak_gpu_allocated_mb']]
    cv_costs=pd.read_csv(OUT/'cv_job_costs.csv').groupby(['branch','mode'],sort=False).seconds.agg(['mean','min','max']).reset_index()
    text=f'''# Embedding Extraction và Ablation Study — DiaGraMo

## Phạm vi và trạng thái

Đánh giá GAF/ViT-B16 và CWT/EfficientNet-B4; không thực hiện Fusion. Giữ nguyên **258 participant,
nhãn group, Stratified 6-Fold, shuffle=True, seed 42**, cùng fold assignments cũ.
Trạng thái hoàn thành cuối cùng chỉ xác nhận bởi `validation.json`: số liệu, notebook, hình và test đều phải qua.

## Vấn đề của baseline và ba phương án

Kernel gốc `W7[:,c] = sum_RGB(W3)/7` giống nhau ở cả bảy kênh. Với chuẩn hóa chung
`(I−0.449)/0.226`, convolution gốc tương đương encoder RGB nhận trung bình bảy ảnh kênh
lặp lại ở R/G/B. Nó bất biến với hoán vị bảy kênh, nên không giữ danh tính kênh ở lớp đầu.
Unit test kiểm tra đại số; trích xuất kiểm tra số trên hai tác vụ đầu với checkpoint thật và embedding cũ.
`baseline_extraction_audit.json` còn xác nhận tái trích xuất đủ **3.612 tác vụ mỗi nhánh** khớp embedding gốc
trong tolerance số học (atol=rtol=3e-4), không ghi đè ma trận gốc. Điều này kiểm tra sự tương thích
của ảnh/cache, thứ tự hàng và checkpoint cho toàn cohort, ngoài kiểm chứng đại số trên hai tác vụ.

- `original`: embedding gốc, dự đoán OOF/inner scores được giữ và đọc lại nguyên vẹn.
- `group_adapter`: 1×1 convolution cố định 7→3; R=mean(x,y), G=mean(pressure,pen_status),
  B=mean(tilt,azimuth,time). Mỗi input tham gia; các hàng là convex average. Phép gộp này vẫn mất
  thông tin và việc gán nhóm vào RGB là giả định cố định, chưa phải ánh xạ tối ưu đã học.
- `channel_mean`: encode riêng từng ảnh kênh bằng encoder RGB không sửa kernel, rồi trung bình bảy
  embedding sau nonlinear encoder. Số chiều bằng baseline, chi phí bảy lượt encoder.
- `channel_concat`: cùng bảy embedding trên nhưng nối theo thứ tự x/y/pressure/tilt/azimuth/time/pen_status.
  Giữ danh tính kênh; số chiều gấp bảy. Thứ tự vẫn là participant → task → channel → embedding coordinate.

Tất cả encoder **đóng băng, eval(), float32**; adapter không có tham số trainable và không đọc nhãn.
Không có adapter học nhãn trên toàn cohort. Cache chỉ chứa embedding label-free; không fine-tune encoder.
Giữ nguyên ảnh 224×224, định nghĩa GAF/CWT và scalar normalization cũ để cô lập channel adaptation.
Đây không phải thử nghiệm chuẩn hóa ImageNet RGB hay tối ưu resolution mới.

## Protocol ablation và chống leakage

Lưới chung: **192 cấu hình** = 2 nhánh × 4 embedding × 8 reducer × 3 classifier.
Reducer: `none`, PCA-16/32, PCA-95%/99%, ANOVA-64/256 và ExtraTrees-256.
Classifier: Logistic Regression, SVM, LightGBM, nguyên siêu tham số baseline.
PCA/selector dùng `ReductionBank` của nghiên cứu trước: imputer → loại cột hằng → StandardScaler → reducer,
tất cả fit chỉ trên train của split tương ứng. PCA không whitening hoặc scale lại sau projection.

Mỗi outer fold có inner Stratified 3-Fold mới trên 215 người train; toàn bộ transform được fit lại
trong inner train. Hàm chọn chỉ nhận inner scores, xếp theo F1 macro rồi Accuracy, AUC, ít chiều hơn và tên.
Không chọn threshold từ test. `original_raw` dùng inner CV chọn classifier; `original_compact` giữ toàn bộ
54 ứng viên cũ (18 reducer kể cả none × 3 classifier), dự đoán nested khớp chính xác đợt trước.
Mỗi embedding mới có 24 ứng viên; `new_selected` chọn giữa 72 ứng viên mới;
`all_selected` chọn giữa 126 ứng viên mới+cũ, cho phép baseline thắng. Pool chọn có kích thước khác nhau,
nên không diễn giải chênh lệch như tác động thuần của số lượng ứng viên.

## Bảng ablation cấu hình cố định — thăm dò

Mỗi hàng chọn classifier/reducer theo chính outer grid, **có selection bias**. Nhóm “có” loại `none`.
Đơn vị %, bộ ba **mean Accuracy / mean F1 macro / mean ROC-AUC của 6 fold**.
SD và pooled OOF được giữ trong `summary.csv`; không trộn hai cách tổng hợp.

{table(pd.DataFrame(fixed))}

Tác động riêng tại **cùng classifier và cùng reducer**: `ablation_matched.csv` có
`delta_adaptation_*_pp` so với embedding gốc cùng reducer; `delta_reduction_*_pp` so với
cùng embedding, giữ nguyên chiều. Đây là đối chiếu đủ ô, không chỉ đối chiếu các hàng thắng khác classifier.
`channel_mean` so với `channel_concat` cô lập giữ danh tính kênh sau encoder với cùng số lượt inference;
số chiều cũng thay đổi nên cần xem thêm các reducer cùng ngân sách.

## Nested CV — ưu tiên để đánh giá lợi ích thực tế

Mọi lựa chọn bên trong train. `raw` chỉ chọn classifier; `compact` cho phép chọn `none`.
Các delta là điểm phần trăm F1 so với nested gốc và nested giảm chiều cũ, cùng outer participant.

{table(pd.DataFrame(nested_table))}

## Chi phí tính toán

Benchmark chung: 32 tác vụ đầu, không dùng nhãn, batch 16, warmup một lần/mode, lặp ba lần,
báo median. Bao gồm transfer CPU↔GPU, encoder và aggregation; **không gồm** tạo ảnh, load model,
fit reducer hoặc classifier. ms/participant là ước tính tuyến tính từ throughput batch × 14 tác vụ,
không phải latency thực đo của một participant đơn lẻ. Bộ nhớ là peak allocated của PyTorch,
không bao gồm toàn bộ VRAM/driver; hai encoder RGB/7ch cùng được load cho benchmark.

{table(costs)}

Thời gian trích xuất toàn cohort theo từng phần ở `embeddings/GAF.json`, `embeddings/CWT.json`.
`channel_mean` và `channel_concat` dùng chung computation, không cộng hai thời gian này như hai extraction độc lập.
`fit_costs.csv` ghi thời gian fit/transform bank từng inner/outer split; `fold_scores.csv` ghi classifier fit và predict.
Bank được tái sử dụng trên các budget trong cùng split, nên thời gian bank không được nhân lặp cho mỗi classifier.
Baseline OOF được đọc lại, chi phí fit bằng 0 ở lần chạy này không có nghĩa baseline không cần train khi dùng thực tế.
Thời gian classifier fit của `original` để trống trong summary vì artifact cũ không tách riêng fit/predict.
Các thời gian CV mới là wall-clock khi chạy tối đa hai fold mỗi nhánh; hai nhánh có thể chạy đồng thời,
chịu ảnh hưởng tranh chấp CPU. Chỉ benchmark GPU chung phía trên là đối chiếu extraction theo cùng workload.

Thời gian mỗi outer job dưới đây bao gồm **toàn bộ 24 ứng viên × (3 inner split + 1 outer fit)**,
không phải thời gian train một cấu hình được chọn. Bảng không tính lại chi phí cũ đã reuse;
không cộng thời gian các job chạy song song để gọi là tổng wall-clock thực nghiệm. Đơn vị giây.

{table(cv_costs)}

## Kết luận và ứng viên cho bước Fusion sau này

{chr(10).join(decisions)}

### Lợi ích đến từ adaptation hay giảm chiều?

{chr(10).join(interpretation)}

Với GAF, mức tăng lớn đã xuất hiện ở concat **chưa giảm chiều**; nén chủ yếu giảm kích thước,
trong khi F1 của hai quy trình concat raw/compact gần bằng nhau và AUC có đánh đổi.
Điều này hỗ trợ hướng sửa phép trộn kênh, không phải kết luận PCA luôn làm tăng hiệu năng.
Với CWT, không phương án mới raw/compact nào vượt quy trình gốc về mean F1 trong nested CV.
`all_selected` của CWT cũng kém baseline dù pool có chứa baseline: cho phép `none` thắng
không bảo đảm lựa chọn từ inner CV sẽ đạt điểm outer test cao hơn. Đây là biểu hiện cần lưu ý
của tính bất định khi chọn mô hình trên cohort nhỏ, chưa phải bằng chứng xác định một nguyên nhân duy nhất.

GAF nối riêng kênh là họ đáng ưu tiên cho ablation Fusion tiếp theo; nó được inner CV chọn ở 5/6 fold
trong pool embedding mới. Giữ bản compact theo train fold để giảm kích thước classifier.
GAF adapter theo nhóm là phương án thay thế chi phí encoder gần baseline, nhưng mức F1 thấp hơn.
GAF channel_mean nên giữ như đối chứng cùng chi phí bảy lượt encoder.
CWT giữ embedding gốc làm mốc chính; các bản mới chỉ giữ để truy nguyên/đối chứng, chưa promote.
Các khác biệt GAF/CWT có thể liên quan đến cấu trúc ảnh và độ dư thừa sau chuẩn hóa riêng từng kênh,
nhưng đây mới là giả thuyết cần kiểm chứng, không suy ra nguyên nhân chỉ từ bảng điểm.

Để chọn embedding cố định cho nghiên cứu Fusion tiếp theo, xem thêm `selection_frequency.csv` về mức ổn định
và `extraction_costs.csv` về đánh đổi tài nguyên. `fusion_candidates.csv` phân loại **họ embedding**:
chỉ có Δ F1 nested dương so với gốc mới được ghi là ứng viên cho ablation Fusion sau này, không phải
xác nhận embedding đó cải thiện Fusion. Tần suất chọn lấy từ inner CV của sáu outer fold.

{table(candidate_table)}

Các cấu hình đứng đầu outer grid chỉ là ứng viên ablation;
nếu lợi ích nested không dương, giữ embedding cũ làm đối chứng thay vì tuyên bố cải tiến.
Không chạy hoặc triển khai Fusion trong nhiệm vụ này.
Ở nghiên cứu Fusion sau, chỉ tái sử dụng cache embedding label-free. PCA/selector phải được fit lại
trong train của split Fusion tương ứng; không lấy tập đặc trưng hợp từ cả sáu outer fold rồi
đánh giá lại trên chính cohort đó như một phép kiểm tra độc lập.

Giữ nguyên baseline Kinematic đầy đủ và MOMENT của đợt trước, không diễn giải việc thay embedding ảnh
là bằng chứng vượt mốc Kinematic. Chênh lệch nhỏ trên cùng 258 người chưa chứng minh ý nghĩa thống kê
hay cải thiện trên external cohort. Đợt thiết kế này tham khảo kết quả cũ trên cùng cohort, nên không
coi đây là một tập xác nhận hoàn toàn độc lập. Sáu outer test fold tách participant nhưng train chồng lấp;
SD fold không phải CI, không dùng t-test độc lập trên sáu fold.

## Artifacts và chạy lại

- `settings.json`, `evaluation_settings.json`, `original_artifacts.json`, `validation.json`.
- `embeddings/`: ma trận mới, checkpoint tiến độ, hash, channel order, chi phí và kiểm chứng baseline.
- `summary.csv`, `fold_scores.csv`, `ablation_matched.csv`, `previous_grid_winners.csv`.
- `nested_summary.csv`, `nested_comparison.csv`, `nested_oof.csv`, `selected_by_fold.csv`, `selection_frequency.csv`.
- `folds/`: train/test IDs, audit từng inner fit, selected feature indices, variance PCA, scores và predictions.
- `selections/`: toàn bộ inner ranking và cấu hình đã chọn theo outer fold.
- `benchmark.json`, `extraction_costs.csv`, `fit_costs.csv`, `cv_job_costs.csv`, `figures/` PNG/SVG và provenance.
- Notebook `notebooks/07_embedding_ablation.ipynb`: bảng số đầy đủ và biểu đồ, không fit mô hình khi mở.

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/extract_embedding_ablation.py --branch GAF
.venv/Scripts/python.exe -X utf8 -B -u scripts/extract_embedding_ablation.py --branch CWT
.venv/Scripts/python.exe -X utf8 -B -u scripts/evaluate_embedding_ablation.py
.venv/Scripts/python.exe -X utf8 -B -u scripts/benchmark_embedding_ablation.py
.venv/Scripts/python.exe -X utf8 -B -u scripts/audit_embedding_baseline.py
.venv/Scripts/python.exe -X utf8 -B scripts/report_embedding_ablation.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_embedding_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 07_embedding_ablation.ipynb --validation-output results/embedding_ablation/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_embedding_ablation.py
```

Extraction và evaluation có checkpoint; thay code/protocol/input sẽ bị từ chối thay vì trộn cache.
Mã bổ sung sử dụng cùng workspace; kết quả/checkpoint/OOF cũ được xác minh SHA-256 không thay đổi.
Tham khảo chính thức: [ViT-B16](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.vit_b_16.html),
[EfficientNet-B4](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.efficientnet_b4.html),
[cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html).
'''
    (OUT/'REPORT_VI.md').write_text(text,encoding='utf-8')


if __name__=='__main__': generate_report()
