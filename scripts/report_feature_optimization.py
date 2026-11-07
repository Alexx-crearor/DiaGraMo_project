"""Validate and report the fold-local reduction study without altering old results."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import metrics
from diagramo.reduction import choose_candidate
from optimize_features import ROOT,BRANCHES,CLASSIFIERS,SPECS,sha,write_json

PRIMARY=('accuracy','f1_macro','auc')


def markdown(df):
    lines=['| '+' | '.join(map(str,df.columns))+' |',
           '| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.2f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def aggregate_oof(d,fold_rows):
    values=metrics(d.label,d.prediction,d.probability)
    for m in values.copy():
        values['mean_'+m]=float(np.mean([r[m] for r in fold_rows]))
        values['std_'+m]=float(np.std([r[m] for r in fold_rows],ddof=1))
    return values


def validate_fold(r,ids,y,folds,settings_hash):
    f=r['fold']; train=ids[folds.fold.to_numpy()!=f].tolist(); test=ids[folds.fold.to_numpy()==f].tolist()
    if r['settings_sha256']!=settings_hash: raise ValueError('Study settings changed')
    if r['train_participants']!=train or r['test_participants']!=test or r['bank']['fit_participants']!=train:
        raise ValueError('Outer training fit audit mismatch')
    if set(train)&set(test): raise ValueError('Participant leakage in outer fold')
    if r['test_indices']!=np.flatnonzero(folds.fold.to_numpy()==f).tolist(): raise ValueError('Test indices mismatch')
    expected={(s['name'],c) for s in SPECS for c in CLASSIFIERS}
    if len(r['scores'])!=len(expected) or {(x['reducer'],x['classifier']) for x in r['scores']}!=expected:
        raise ValueError('Missing/duplicate outer candidate')
    validation_ids=[]
    for audit in r['selection']['audits']:
        tr=audit['train_participants']; va=audit['validation_participants']
        if audit['bank']['fit_participants']!=tr or set(tr)&set(va) or set(tr)|set(va)!=set(train):
            raise ValueError('Inner transform leakage / partition mismatch')
        if (set(tr)|set(va))&set(test): raise ValueError('Outer test participant reached inner CV')
        validation_ids.extend(va)
    if sorted(validation_ids)!=sorted(train): raise ValueError('Inner validation coverage mismatch')
    inner=pd.DataFrame(r['selection']['inner_fold_scores'])
    recomputed=[]
    for (reducer,clf),g in inner.groupby(['reducer','classifier'],sort=False):
        if len(g)!=3 or set(g.inner_fold)!={0,1,2}: raise ValueError('Incomplete inner CV grid')
        recomputed.append(dict(reducer=reducer,classifier=clf,mean_features=float(g.n_features.mean()),
                               **{'mean_'+m:float(g[m].mean()) for m in PRIMARY}))
    if len(recomputed)!=len(expected): raise ValueError('Inner candidate coverage mismatch')
    for scenario,candidates in [('optimized',recomputed),('baseline',[x for x in recomputed if x['reducer']=='none'])]:
        if choose_candidate(candidates)!=r['selection'][scenario]:
            raise ValueError('Selected configuration differs from training-only ranking')
    for score in r['scores']:
        metadata=r['reduction_metadata'][score['reducer']]
        if score['n_features']!=metadata['n_features']:
            raise ValueError('Reported classifier width differs from fitted reduction')
        if metadata['method'] in ('anova','extratrees'):
            indices=metadata['selected_original_indices']
            if (len(indices)!=score['n_features'] or len(set(indices))!=len(indices)
                    or any(i<0 or i>=r['bank']['raw_features'] for i in indices)):
                raise ValueError('Invalid selected original feature indices')
        key=score['reducer']+'__'+score['classifier']; p=r['predictions'][key]
        pred=np.asarray(p['prediction']); prob=np.asarray(p['probability'])
        if pred.shape!=(len(test),) or prob.shape!=(len(test),) or not np.isin(pred,[0,1]).all():
            raise ValueError('Invalid outer predictions')
        if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)): raise ValueError('Invalid probabilities')
        truth=y[folds.fold.to_numpy()==f]
        for m,v in metrics(truth,pred,prob).items():
            if not np.isclose(v,score[m],atol=1e-12,rtol=0): raise ValueError('Fold metrics do not match predictions')


def generate_report(out):
    config=json.loads((out/'settings.json').read_text(encoding='utf-8'))
    settings_hash=sha(out/'settings.json')
    folds=pd.read_csv(out/'fold_assignments.csv'); ids=folds.participant_id.to_numpy(); y=folds.label.to_numpy()
    expected={(b,f) for b in BRANCHES for f in range(6)}
    results=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'folds').glob('*.json'))]
    if len(results)!=24 or {(r['branch'],r['fold']) for r in results}!=expected:
        raise ValueError('All 24 outer jobs must finish before reporting complete results')
    for r in results: validate_fold(r,ids,y,folds,settings_hash)
    preserved=json.loads((out/'original_artifacts.json').read_text(encoding='utf-8'))
    changed=[p for p,h in preserved.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    if changed: raise ValueError('Original artifacts changed: '+str(changed))
    for key,h in config['input_sha256'].items():
        if sha(ROOT/'data/derived'/(key+'.npz'))!=h: raise ValueError('Input features changed')
    if sha(ROOT/'results/cv/fold_assignments.csv')!=config['fold_assignments_sha256']:
        raise ValueError('Original folds changed')
    for p,h in config['code_sha256'].items():
        if sha(ROOT/p)!=h: raise ValueError('Experiment code changed during the run')

    oof_dir=out/'oof'; oof_dir.mkdir(exist_ok=True)
    rows=[]; fold_rows=[]; nested_rows=[]; selected=[]; nested_oof=[]; stability=[]
    for branch,key in BRANCHES.items():
        six=sorted([r for r in results if r['branch']==branch],key=lambda r:r['fold'])
        raw=six[0]['bank']['raw_features']
        for spec in SPECS:
            dims=[r['reduction_metadata'][spec['name']]['n_features'] for r in six]
            if spec['method'] in ('anova','extratrees'):
                supports=[set(r['reduction_metadata'][spec['name']]['selected_original_indices']) for r in six]
                jaccard=[len(a&b)/len(a|b) for a,b in itertools.combinations(supports,2)]
                stability.append(dict(branch=branch,reducer=spec['name'],mean_pairwise_jaccard=float(np.mean(jaccard)),
                                      n_features=float(np.mean(dims))))
            for clf in CLASSIFIERS:
                d=folds.copy(); d['prediction']=np.zeros(len(ids),dtype=int); d['probability']=np.nan
                scores=[]
                for r in six:
                    score=next(s for s in r['scores'] if s['reducer']==spec['name'] and s['classifier']==clf)
                    scores.append(score)
                    fold_rows.append(dict(branch=branch,representation=key,fold=r['fold'],**score))
                    p=r['predictions'][spec['name']+'__'+clf]; ix=r['test_indices']
                    d.loc[ix,'prediction']=p['prediction']; d.loc[ix,'probability']=p['probability']
                if d.probability.isna().any() or not d.participant_id.is_unique: raise ValueError('OOF coverage mismatch')
                if spec['method']=='none':
                    old=pd.read_csv(ROOT/'results/cv'/f'{key}__{clf}.oof.csv',float_precision='round_trip')
                    np.testing.assert_array_equal(d.prediction,old.prediction)
                    np.testing.assert_array_equal(d.probability,old.probability)
                d.to_csv(oof_dir/f'{branch}__{spec["name"]}__{clf}.oof.csv',index=False)
                ev=[r['reduction_metadata'][spec['name']].get('explained_variance',np.nan) for r in six]
                rows.append(dict(branch=branch,representation=key,reducer=spec['name'],method=spec['method'],classifier=clf,
                                 raw_features=raw,mean_features=float(np.mean(dims)),min_features=min(dims),max_features=max(dims),
                                 reduction_pct=100*(1-float(np.mean(dims))/raw),
                                 mean_explained_variance=float(np.mean(ev)),
                                 mean_classifier_seconds=float(np.mean([s['seconds'] for s in scores])),
                                 **aggregate_oof(d,scores)))
        for scenario in ('baseline','optimized'):
            d=folds.copy(); d['prediction']=np.zeros(len(ids),dtype=int); d['probability']=np.nan
            scores=[]; dimensions=[]
            for r in six:
                choice=r['selection'][scenario]; spec_name=choice['reducer']; clf=choice['classifier']
                p=r['predictions'][spec_name+'__'+clf]; ix=r['test_indices']
                d.loc[ix,'prediction']=p['prediction']; d.loc[ix,'probability']=p['probability']
                score=next(s for s in r['scores'] if s['reducer']==spec_name and s['classifier']==clf)
                scores.append(score); dimensions.append(score['n_features'])
                selected.append(dict(branch=branch,scenario=scenario,fold=r['fold'],
                                     **choice,n_features=score['n_features'],
                                     **{'outer_'+m:score[m] for m in PRIMARY}))
            nested_oof.append(d.assign(branch=branch,scenario=scenario))
            nested_rows.append(dict(branch=branch,scenario=scenario,raw_features=raw,
                                    mean_features=float(np.mean(dimensions)),**aggregate_oof(d,scores)))
    summary=pd.DataFrame(rows).sort_values(['branch','mean_f1_macro','mean_accuracy','mean_auc'],ascending=[True,False,False,False])
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_scores.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'feature_stability.csv',index=False)
    selected=pd.DataFrame(selected); selected.to_csv(out/'selected_by_fold.csv',index=False)
    pd.concat(nested_oof,ignore_index=True).to_csv(out/'nested_oof.csv',index=False)
    nested=pd.DataFrame(nested_rows); nested.to_csv(out/'nested_summary.csv',index=False)
    comparison=summary[summary.method!='none'].copy()
    baseline=summary[summary.method=='none'].set_index(['branch','classifier'])
    for m in PRIMARY:
        comparison['baseline_mean_'+m]=[baseline.loc[(r.branch,r.classifier),'mean_'+m] for r in comparison.itertuples()]
        comparison['delta_'+m+'_pp']=100*(comparison['mean_'+m]-comparison['baseline_mean_'+m])
    comparison.to_csv(out/'comparison_same_classifier.csv',index=False)
    best=comparison.groupby('branch',sort=False).head(1).copy()
    # Keep strongest existing classifier as an additional, separate comparator.
    old_summary=pd.read_csv(ROOT/'results/cv/summary.csv')
    old_best=[]
    for branch,key in BRANCHES.items():
        g=old_summary[(old_summary.representation==key)&(old_summary.classifier!='Dummy')]
        b=g.sort_values(['mean_f1_macro','mean_accuracy','mean_auc'],ascending=False).iloc[0]
        old_best.append(dict(branch=branch,classifier=b.classifier,**{'mean_'+m:b['mean_'+m] for m in PRIMARY}))
        mask=best.branch==branch
        best.loc[mask,'old_best_classifier']=b.classifier
        for m in PRIMARY:
            best.loc[mask,'old_best_mean_'+m]=b['mean_'+m]
            best.loc[mask,'delta_vs_old_best_'+m+'_pp']=100*(best.loc[mask,'mean_'+m]-b['mean_'+m])
    best.to_csv(out/'best_per_branch.csv',index=False)
    by_method=comparison.groupby(['branch','method'],sort=False).head(1)
    by_method.to_csv(out/'best_by_method.csv',index=False)
    nested_compare=[]
    for branch in BRANCHES:
        a=nested[(nested.branch==branch)&(nested.scenario=='baseline')].iloc[0]
        b=nested[(nested.branch==branch)&(nested.scenario=='optimized')].iloc[0]
        nested_compare.append(dict(branch=branch,mean_features=b.mean_features,
                                   **{'baseline_'+m:a['mean_'+m] for m in PRIMARY},
                                   **{'optimized_'+m:b['mean_'+m] for m in PRIMARY},
                                   **{'delta_'+m+'_pp':100*(b['mean_'+m]-a['mean_'+m]) for m in PRIMARY}))
    nested_compare=pd.DataFrame(nested_compare); nested_compare.to_csv(out/'nested_comparison.csv',index=False)
    write_report(out,config,summary,comparison,best,by_method,nested_compare,selected,pd.DataFrame(old_best))
    write_json(out/'validation.json',dict(complete=True,expected_grid_results=216,actual_grid_results=len(summary),
                                        outer_jobs_validated=len(results),inner_fit_partitions_validated=72,
                                        fixed_oof_files_validated=len(summary),nested_oof_subject_rows=8*len(ids),
                                        original_artifacts_verified_unchanged=len(preserved),
                                        old_results_unchanged=True,feature_inputs_unchanged=True,
                                        baseline_oof_reuse_exact=True,outer_folds_match_original=True,
                                        selection_uses_only_inner_cv=True))
    print('VALIDATED:',len(summary),'grid results, 24 outer folds, 72 inner partitions; original artifacts unchanged.',flush=True)


def write_report(out,config,summary,comparison,best,by_method,nested_compare,selected,old_best):
    def triplet(row,prefix='mean_'):
        return ' / '.join(f'{100*row[prefix+m]:.2f}' for m in PRIMARY)
    best_table=[]
    method_table=[]
    for branch in BRANCHES:
        r=best[best.branch==branch].iloc[0]; old=old_best[old_best.branch==branch].iloc[0]
        best_table.append({'Nhánh':branch,'Cấu hình giảm chiều':r.reducer+' + '+r.classifier,
                           'Chiều gốc → mới':f'{int(r.raw_features)} → {r.mean_features:.1f}',
                           'Baseline tốt nhất cũ (Acc/F1/AUC)':old.classifier+': '+triplet(old),
                           'Mới (Acc/F1/AUC)':triplet(r),
                           'Δ F1 so với baseline cũ (đpt)':r.delta_vs_old_best_f1_macro_pp})
        for row in by_method[by_method.branch==branch].itertuples():
            method_table.append({'Nhánh':branch,'Phương pháp':row.method,'Cấu hình':row.reducer+' + '+row.classifier,
                                 'Acc':100*row.mean_accuracy,'F1 macro':100*row.mean_f1_macro,'AUC':100*row.mean_auc})
    nested_table=[]
    for r in nested_compare.itertuples():
        nested_table.append({'Nhánh':r.branch,
                             'Chọn classifier, giữ nguyên chiều (Acc/F1/AUC)':
                             ' / '.join(f'{100*getattr(r,"baseline_"+m):.2f}' for m in PRIMARY),
                             'Chọn cả reducer và classifier (Acc/F1/AUC)':
                             ' / '.join(f'{100*getattr(r,"optimized_"+m):.2f}' for m in PRIMARY),
                             'Δ F1 (đpt)':r.delta_f1_macro_pp,'Chiều TB sau chọn':r.mean_features})
    choices=selected[selected.scenario=='optimized'].groupby(['branch','reducer','classifier'],sort=True).size().reset_index(name='outer_folds_selected')
    choices.to_csv(out/'selection_frequency.csv',index=False)
    commentary=[]
    for branch in BRANCHES:
        r=best[best.branch==branch].iloc[0]; n=nested_compare[nested_compare.branch==branch].iloc[0]
        supported=('cải thiện F1 trong đánh giá nested' if n.delta_f1_macro_pp>0
                   else 'không cải thiện F1 trong đánh giá nested' if n.delta_f1_macro_pp<0
                   else 'giữ nguyên F1 trong đánh giá nested')
        commentary.append(f'- **{branch}:** cấu hình giảm chiều đứng đầu bảng thăm dò là `{r.reducer} / {r.classifier}` '
                          f'({r.mean_features:.1f}/{int(r.raw_features)} chiều, giảm {r.reduction_pct:.2f}%). '
                          f'F1 macro {100*r.mean_f1_macro:.2f}% (SD {100*r.std_f1_macro:.2f} điểm); '
                          f'chênh baseline cùng classifier {r.delta_f1_macro_pp:+.2f} điểm và baseline cũ tốt nhất '
                          f'{r.delta_vs_old_best_f1_macro_pp:+.2f} điểm. Quy trình chọn trong train {supported} '
                          f'({n.delta_f1_macro_pp:+.2f} điểm).')
    method_wins=best.method.value_counts().to_dict()
    text=f'''# Feature Selection và Dimensionality Reduction — DiaGraMo

Đã hoàn thành **216 cấu hình** = 4 nhánh × (17 cấu hình giảm chiều + baseline) × 3 classifier,
trên cùng **258 participant, Stratified 6-Fold, shuffle=True, seed 42**. Trong đó 12 baseline được đọc
nguyên dự đoán OOF cũ; 204 cấu hình giảm chiều được chạy mới. Thêm **nested CV: 6 outer × 3 inner**
để đánh giá quy trình chọn cả phương pháp/mức chiều/classifier bằng dữ liệu train.
Toàn bộ kết quả cũ và 4 notebook cũ được kiểm tra SHA-256, không bị sửa.

## Thiết lập đã cố định trước khi chạy

- Kinematic: 1.036; MOMENT: 14.336; GAF/ViT-B16: 10.752; CWT/EfficientNet-B4: 25.088 chiều.
- PCA full SVD: **16, 32, 64, 128** thành phần; hoặc giữ **90%, 95%, 99%** phương sai train.
  Không whitening và không chuẩn hóa lại các thành phần sau PCA. Vì train outer có 215 người,
  số thành phần có phương sai khác 0 tối đa 214; train inner có 143–144 người.
- SelectKBest ANOVA `f_classif`: **32, 64, 128, 256, 512** đặc trưng.
- SelectFromModel với ExtraTrees: **32, 64, 128, 256, 512** đặc trưng quan trọng nhất;
  200 cây, min_samples_leaf=2, max_features=sqrt, random_state=42. Cây chỉ dùng nhãn train.
- Logistic Regression, SVM RBF và LightGBM dùng nguyên tham số từ baseline `scripts/evaluate.py`.
  Đây là nghiên cứu 3 classifier đại diện; chưa lặp lại Random Forest, XGBoost, AdaBoost và MLP.
- Metric chính để chọn: **mean F1 macro**; tie-break: Accuracy, AUC, ít chiều hơn, tên cấu hình.
  Không chọn ngưỡng dự đoán hoặc thay nhãn/cohort dựa vào điểm test.

## Không rò rỉ dữ liệu

Mỗi split có một `ReductionBank` riêng. Pipeline median imputation → loại cột hằng →
StandardScaler được fit trên train, sau đó PCA/ANOVA/ExtraTrees cũng chỉ fit trên train.
Một PCA full SVD và một bộ điểm selector được tái sử dụng cho các mức chiều/classifier trên
**cùng train split**; đây chỉ là chia sẻ tính toán, không fit trên toàn cohort. Biến đổi của
outer train tuyệt đối không được tái sử dụng cho inner train.

Hàm chọn nội bộ chỉ nhận `X_train, y_train, participant_ids_train`; không nhận outer test.
Mọi audit inner/outer được kiểm tra tập ID train/test tách rời, coverage và việc chọn đúng
theo điểm inner-CV. Với PCA 90/95/99%, số chiều được xác định riêng từ phương sai train.

## Cấu hình cố định tốt nhất trong bảng thăm dò

Đơn vị **%**; mỗi bộ ba là **Accuracy / F1 macro / ROC-AUC**, đều là **trung bình 6 fold**.
`summary.csv` giữ thêm SD và OOF gộp. AUC gộp có thể khác AUC trung bình.
Các cấu hình dưới đây được xếp hạng trên chính 6 fold nên có **thiên lệch chọn cấu hình**;
không xem điểm của cấu hình thắng bảng này là ước lượng độc lập sau tối ưu.
Baseline cũ tốt nhất được chọn theo cùng tiêu chí F1 macro trong toàn bộ 7 classifier cũ.

{markdown(pd.DataFrame(best_table))}

Đối chiếu từng cấu hình với baseline **cùng classifier** nằm ở
[comparison_same_classifier.csv](comparison_same_classifier.csv); không gán lợi ích đổi classifier cho giảm chiều.

## So sánh phương pháp tốt nhất theo nhánh

{markdown(pd.DataFrame(method_table))}

## Kiểm tra bằng nested CV — ưu tiên bảng này để kết luận về tối ưu

Ở mỗi outer fold, inner Stratified 3-Fold trên 215 người train chọn trong **54 ứng viên**
(18 reducer kể cả `none` × 3 classifier). Đối chứng chọn trong 3 classifier giữ nguyên chiều,
cũng chỉ dựa trên inner CV. Hai lựa chọn được đánh giá trên cùng 43 người outer test.
Nhánh `none` được phép thắng nếu giảm chiều không có lợi; không bắt buộc chọn một reducer.
Mỗi outer fold có thể chọn cấu hình khác nhau: bảng này đánh giá **quy trình chọn**, không xác nhận
một cấu hình cố định là tối ưu trên dữ liệu mới. 3-fold inner được dùng thêm, còn giao thức
đánh giá outer 6-fold, seed và participant giữ nguyên baseline.

{markdown(pd.DataFrame(nested_table))}

Lựa chọn từng fold: [selected_by_fold.csv](selected_by_fold.csv).
Tần suất chọn: [selection_frequency.csv](selection_frequency.csv).

## Phương pháp nào tốt nhất và cách giải thích

{chr(10).join(commentary)}

Trong bảng thăm dò, số nhánh đứng đầu theo phương pháp: **{method_wins}**. Đây là mô tả kết quả
trên cohort hiện tại, không phải bằng chứng một phương pháp luôn tốt nhất.

**Khuyến nghị từ đợt chạy này:** giữ baseline đầy đủ chiều của Kinematic nếu ưu tiên F1 macro.
MOMENT là nhánh có bằng chứng mô tả tích cực nhất cho việc tiếp tục tối ưu: ANOVA-64 + Logistic
Regression đứng đầu bảng cố định và quy trình chọn bằng inner CV cũng tăng F1 trên outer test.
Tuy nhiên, inner CV không chọn ANOVA-64 ở mọi fold; mức tăng nested đánh giá toàn bộ quy trình
chọn, chưa xác nhận riêng cấu hình ANOVA-64. Với GAF và CWT, giữ baseline làm cấu hình tham chiếu
cho đến khi có đánh giá xác nhận thêm, vì lợi ích của cấu hình thắng bảng không lặp lại trong nested CV.

Một cách giải thích **cần kiểm chứng** là ANOVA loại nhiều tọa độ MOMENT ít phân biệt hai nhóm,
còn PCA gộp các tọa độ CWT tương quan. Đây là cơ chế phù hợp với phương pháp, chưa phải nguyên nhân
đã chứng minh bằng thực nghiệm ablation. Số chiều rất lớn so với 215 người train và việc inner CV
chọn các cấu hình khác nhau giữa fold cho thấy cần thận trọng với tính ổn định của cấu hình thắng.

- **PCA** loại dư thừa tuyến tính bằng cách xoay sang các trục phương sai lớn. Nó hữu ích khi
  embedding có nhiều tọa độ tương quan, nhưng không dùng nhãn nên có thể bỏ hướng phương sai nhỏ
  chứa tín hiệu phân lớp. Giữ 99% phương sai không đồng nghĩa giữ 99% thông tin chẩn đoán.
- **ANOVA** giữ tọa độ có khác biệt trung bình giữa hai nhóm, dễ diễn giải hơn PCA đối với động học.
  Nó không mô hình hóa tương tác và có thể giữ nhiều biến tương quan. Trong p≫n, điểm univariate
  dễ biến động; vì vậy chỉ chọn trong train và có nested CV.
- **ExtraTrees** xếp hạng theo phân chia cây, có khả năng khai thác quan hệ phi tuyến/tương tác.
  Với chỉ 215 người train và hàng chục nghìn tọa độ, importance có thể phân tán giữa các biến
  tương quan; không diễn giải importance như quan hệ nhân quả hay bằng chứng lâm sàng.
- [feature_stability.csv](feature_stability.csv) ghi Jaccard trung bình giữa 15 cặp outer-fold
  cho mỗi selector/mức chiều. Các tập train chồng lấp nên đây chỉ là chỉ báo độ ổn định mô tả.
- Giảm chiều giảm kích thước đầu vào classifier, **không** giảm số chiều cần trích từ encoder trước
  reducer. Không tuyên bố tăng tốc end-to-end vì chưa benchmark toàn pipeline có cùng điều kiện.

Khác biệt điểm nhỏ trên 258 người chưa chứng minh ý nghĩa thống kê hoặc khả năng tổng quát hóa.
SD giữa 6 fold không phải khoảng tin cậy; không dùng kiểm định độc lập sai cho các train fold chồng lấp.
Các giả định nhãn `group`, bộ đặc trưng dựng lại và encoder của thực nghiệm cũ vẫn giữ nguyên;
chưa có tập kiểm tra ngoài và chưa có đánh giá chẩn đoán lâm sàng.

## Tệp kết quả và chạy lại

- [summary.csv](summary.csv): đủ 216 cấu hình; mean, SD, OOF gộp, số chiều và phương sai PCA.
- [best_per_branch.csv](best_per_branch.csv), [best_by_method.csv](best_by_method.csv).
- [nested_comparison.csv](nested_comparison.csv), [nested_summary.csv](nested_summary.csv),
  [nested_oof.csv](nested_oof.csv): đánh giá quy trình chọn bằng train-CV.
- [fold_scores.csv](fold_scores.csv), [oof/](oof/): điểm và dự đoán từng người của mọi cấu hình cố định.
- [folds/](folds/): fit participant IDs, selected feature indices, phương sai PCA, toàn bộ điểm inner-CV.
- [settings.json](settings.json), [original_artifacts.json](original_artifacts.json),
  [validation.json](validation.json): thiết lập, SHA-256 và kiểm tra tính toàn vẹn.

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/optimize_features.py --workers 2
```

Chạy lại cùng lệnh sẽ tiếp tục từ các outer job đã hoàn tất; nếu đổi tham số, mã thực nghiệm hoặc
đặc trưng, phải dùng `--output results/feature_optimization_<ten_moi>` để không trộn kết quả.
Tạo lại báo cáo từ kết quả đã chạy:

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/optimize_features.py --report-only
```

Tham khảo kỹ thuật: [scikit-learn 1.7 feature selection](https://scikit-learn.org/1.7/modules/feature_selection.html),
[PCA](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.PCA.html),
[nested CV](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
'''
    (out/'REPORT_VI.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/feature_optimization')
    generate_report((ROOT/parser.parse_args().output).resolve())
