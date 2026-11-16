"""Aggregate held-out SHAP, permutation, task ablation and stability results."""
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from kinematic_xai_common import ROOT, OUT, METRICS, load_data, write, metrics
from diagramo.kinematic_xai import aggregate, blocks, prediction_changes

EXP = OUT / 'exploratory'
TABLES = EXP / 'tables'


def save(name, frame):
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / (name + '.csv'), index=False)
    return frame


def load_shap():
    X, y, ids, schema, assignments, oof, splits = load_data()
    phi = np.zeros_like(X, dtype=np.float64)
    base, raw = np.zeros(len(y)), np.zeros(len(y))
    for f, (_, te) in enumerate(splits):
        with np.load(EXP / 'shap' / f'fold_{f}.npz') as d:
            np.testing.assert_array_equal(d['test_indices'], te)
            phi[te], base[te], raw[te] = d['phi'], d['base'], d['raw_score']
    return X, y, ids, assignments.fold.to_numpy(), oof, phi, base, raw


def importance(values, names, n_features, folds, key):
    fold_means = pd.DataFrame(values, columns=names).groupby(folds).mean()
    ranks = fold_means.rank(axis=1, ascending=False, method='average')
    rows = []
    for j, name in enumerate(names):
        rows.append({key: name, 'n_features': int(n_features[j]),
              'mean_absolute_shap_mass': values[:, j].mean(),
              'total_absolute_shap_mass': values[:, j].sum(),
              'mass_per_feature': values[:, j].mean() / n_features[j],
              'share_percent': values[:, j].mean() / values.sum(axis=1).mean() * 100,
              'participant_q25': np.quantile(values[:, j], .25),
              'participant_q75': np.quantile(values[:, j], .75),
              'fold_mass_sd': fold_means[name].std(ddof=1),
              'mean_rank': ranks[name].mean(), 'rank_sd': ranks[name].std(ddof=1),
              'top3_count': int((ranks[name] <= 3).sum()), 'top3_frequency': float((ranks[name] <= 3).mean())})
    frame = pd.DataFrame(rows).sort_values('mean_absolute_shap_mass', ascending=False)
    frame['global_rank'] = frame.mean_absolute_shap_mass.rank(ascending=False, method='average')
    fold_rows = []
    for f in fold_means.index:
        for name in names:
            fold_rows.append(dict(fold=int(f), **{key: name}, mean_absolute_shap_mass=fold_means.loc[f, name], rank=ranks.loc[f, name]))
    return frame, pd.DataFrame(fold_rows), fold_means


def main():
    X, y, ids, folds, oof, phi, base, raw = load_shap()
    mapping = pd.read_csv(OUT / 'feature_mapping.csv')
    all_fold_importance, fold_matrices = {}, {}
    task_mass = aggregate(phi, mapping, 'task')
    task_signed = aggregate(phi, mapping, 'task', absolute=False)
    group_mass = aggregate(phi, mapping, 'group')
    for level in ('feature', 'task', 'group'):
        if level == 'feature':
            names, counts, values = mapping.full_name.tolist(), np.ones(len(mapping), dtype=int), np.abs(phi)
        else:
            grouped = blocks(mapping, level)
            names, counts = list(grouped), [len(v) for v in grouped.values()]
            values = aggregate(phi, mapping, level).to_numpy()
        summary, per_fold, matrix = importance(values, names, counts, folds, level)
        if level == 'feature':
            summary = summary.merge(mapping.rename(columns={'full_name': 'feature_name'}), left_on='feature', right_on='feature_name', suffixes=('_full', '_source'))
            summary = summary.rename(columns={'feature_full': 'full_name', 'feature_source': 'feature'})
            summary = summary.drop(columns='feature_name')
            save('top20_features', summary.head(20))
        save(level + '_importance', summary)
        save(level + '_importance_by_fold', per_fold)
        all_fold_importance[level] = per_fold
        fold_matrices[level] = matrix
    feature_groups = mapping.groupby(['task', 'group'], sort=False).column.apply(list)
    task_group_rows = []
    for (task, group), columns in feature_groups.items():
        task_group_rows.append(dict(task=task, group=group, n_features=len(columns),
              mean_absolute_shap_mass=np.abs(phi[:, columns]).sum(axis=1).mean(),
              mass_per_feature=np.abs(phi[:, columns]).mean()))
    save('task_group_matrix', pd.DataFrame(task_group_rows))
    participant_rows = []
    participant_ranks = task_mass.rank(axis=1, ascending=False, method='average')
    for i, pid in enumerate(ids):
        for task in task_mass.columns:
            participant_rows.append(dict(participant_id=pid, fold=int(folds[i]), label=int(y[i]),
                prediction=int(oof.prediction[i]), correct=bool(y[i] == oof.prediction[i]), task=task,
                absolute_shap_mass=task_mass.loc[i, task], signed_shap_log_odds=task_signed.loc[i, task],
                share_percent=100 * task_mass.loc[i, task] / task_mass.loc[i].sum(), rank=participant_ranks.loc[i, task]))
    participants = save('participant_task_attributions', pd.DataFrame(participant_rows))
    variability = participants.groupby('task', sort=False).agg(
        participant_rank_mean=('rank', 'mean'), participant_rank_sd=('rank', 'std'),
        participant_top3_frequency=('rank', lambda r: (r <= 3).mean()),
        mass_sd=('absolute_shap_mass', 'std'), mass_min=('absolute_shap_mass', 'min'), mass_max=('absolute_shap_mass', 'max')).reset_index()
    save('participant_variability', variability)
    save('task_signed_summary', pd.DataFrame(dict(task=task_signed.columns,
         mean_signed_log_odds=task_signed.mean().to_numpy(),
         mean_absolute_net_log_odds=task_signed.abs().mean().to_numpy(),
         mean_absolute_feature_mass=task_mass.mean().to_numpy())))
    correctness = participants.groupby(['correct', 'task'], sort=False).agg(
        n=('participant_id', 'size'), mean_absolute_shap_mass=('absolute_shap_mass', 'mean'),
        median_absolute_shap_mass=('absolute_shap_mass', 'median'), mean_signed_log_odds=('signed_shap_log_odds', 'mean'),
        top3_frequency=('rank', lambda r: (r <= 3).mean())).reset_index()
    save('correct_vs_incorrect_tasks', correctness)
    stability = []
    for level, matrix in fold_matrices.items():
        for a, b in itertools.combinations(range(6), 2):
            for k in ((20,) if level == 'feature' else (3, 5)):
                top_a = set(matrix.loc[a].nlargest(k).index)
                top_b = set(matrix.loc[b].nlargest(k).index)
                stability.append(dict(level=level, fold_a=a, fold_b=b, top_k=k,
                    spearman=float(spearmanr(matrix.loc[a], matrix.loc[b]).statistic),
                    overlap_count=len(top_a & top_b), jaccard=len(top_a & top_b) / len(top_a | top_b)))
    stable = save('fold_stability', pd.DataFrame(stability))
    save('fold_stability_summary', stable.groupby(['level', 'top_k']).agg(
        mean_spearman=('spearman', 'mean'), min_spearman=('spearman', 'min'), max_spearman=('spearman', 'max'),
        mean_jaccard=('jaccard', 'mean'), mean_overlap=('overlap_count', 'mean')).reset_index())
    participant_stability = []
    for i, pid in enumerate(ids):
        participant_stability.append(dict(participant_id=pid, fold=int(folds[i]), correct=bool(y[i] == oof.prediction[i]),
           spearman_to_global=float(spearmanr(task_mass.iloc[i], task_mass.mean()).statistic),
           top3_jaccard_to_global=len(set(task_mass.iloc[i].nlargest(3).index) & set(task_mass.mean().nlargest(3).index)) /
               len(set(task_mass.iloc[i].nlargest(3).index) | set(task_mass.mean().nlargest(3).index))))
    save('participant_rank_stability', pd.DataFrame(participant_stability))
    permutation = pd.concat([pd.read_csv(EXP / 'permutation' / f'fold_{f}.csv') for f in range(6)], ignore_index=True)
    save('permutation_repeats', permutation)
    columns = ['drop_' + k for k in METRICS]
    perm_fold = permutation.groupby(['level', 'block', 'fold'], sort=False)[columns].mean().reset_index()
    save('permutation_by_fold', perm_fold)
    permutation_summary = perm_fold.groupby(['level', 'block'], sort=False)[columns].agg(['mean', 'std']).reset_index()
    permutation_summary.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c for c in permutation_summary.columns]
    save('permutation_importance', permutation_summary)
    ablation = pd.concat([pd.read_csv(EXP / 'ablation' / f'fold_{f}.oof.csv', float_precision='round_trip') for f in range(6)], ignore_index=True)
    ablation_folds = pd.concat([pd.read_csv(EXP / 'ablation' / f'fold_{f}.csv') for f in range(6)], ignore_index=True)
    save('task_ablation_oof', ablation)
    save('task_ablation_by_fold', ablation_folds)
    baseline = {k: v for k, v in metrics(y, oof.prediction, oof.probability).items() if k in METRICS}
    baseline_fold_rows = []
    for f in range(6):
        mask = folds == f
        baseline_fold_rows.append({k: v for k, v in metrics(y[mask], oof.prediction[mask], oof.probability[mask]).items() if k in METRICS})
    baseline_fold_frame = pd.DataFrame(baseline_fold_rows)
    rows = [dict(method='original_baseline', task='all_14', **baseline,
                 **{'mean_' + k: baseline_fold_frame[k].mean() for k in METRICS},
                 **{'std_' + k: baseline_fold_frame[k].std(ddof=1) for k in METRICS})]
    for (method, task), d in ablation.groupby(['method', 'task'], sort=False):
        df = ablation_folds[(ablation_folds.method == method) & (ablation_folds.task == task)]
        value = {k: v for k, v in metrics(d.label, d.prediction, d.probability).items() if k in METRICS}
        rows.append(dict(method=method, task=task, **value,
            **{'delta_' + k: value[k] - baseline[k] for k in METRICS},
            **{'mean_' + k: df[k].mean() for k in METRICS},
            **{'std_' + k: df[k].std(ddof=1) for k in METRICS},
            **{'mean_delta_' + k: df['delta_' + k].mean() for k in METRICS},
            **{'std_delta_' + k: df['delta_' + k].std(ddof=1) for k in METRICS},
            corrected=int((d.change == 'corrected').sum()), harmed=int((d.change == 'harmed').sum()),
            unchanged_correct=int((d.change == 'unchanged_correct').sum()), unchanged_wrong=int((d.change == 'unchanged_wrong').sum())))
    ablation_summary = save('task_ablation_summary', pd.DataFrame(rows))
    task_summary = pd.read_csv(TABLES / 'task_importance.csv')
    group_summary = pd.read_csv(TABLES / 'group_importance.csv')
    agreement_rows, concordance = [], []
    for level, summary in [('task', task_summary), ('group', group_summary)]:
        summary = summary.set_index(level)
        perm = permutation_summary[permutation_summary.level == level].set_index('block').loc[summary.index]
        for metric in METRICS:
            concordance.append(dict(level=level, comparison='SHAP_vs_permutation', metric=metric,
                 n_blocks=len(summary), spearman=float(spearmanr(summary.mean_absolute_shap_mass, perm['drop_' + metric + '_mean']).statistic)))
        if level == 'task':
            for task in summary.index:
                row = dict(task=task, shap_mass=summary.loc[task, 'mean_absolute_shap_mass'], shap_rank=summary.loc[task, 'global_rank'],
                     permutation_f1_drop=perm.loc[task, 'drop_f1_macro_mean'])
                for method in ('test_time_train_median', 'leave_one_task_out_refit'):
                    r = ablation_summary[(ablation_summary.method == method) & (ablation_summary.task == task)].iloc[0]
                    row[method + '_f1_drop'] = -r.delta_f1_macro
                agreement_rows.append(row)
            for method in ('test_time_train_median', 'leave_one_task_out_refit'):
                ab = ablation_summary[ablation_summary.method == method].set_index('task').loc[summary.index]
                for metric in METRICS:
                    concordance.append(dict(level='task', comparison='SHAP_vs_' + method, metric=metric,
                         n_blocks=len(summary), spearman=float(spearmanr(summary.mean_absolute_shap_mass, -ab['delta_' + metric]).statistic)))
    agreement = pd.DataFrame(agreement_rows)
    for key in ('permutation', 'test_time_train_median', 'leave_one_task_out_refit'):
        agreement[key + '_rank'] = agreement[key + '_f1_drop'].rank(ascending=False, method='average')
    agreement['shap_loto_rank_gap'] = (agreement.shap_rank - agreement.leave_one_task_out_refit_rank).abs()
    agreement['top3_shap_and_permutation'] = (agreement.shap_rank <= 3) & (agreement.permutation_rank <= 3)
    agreement['top3_all_three_methods'] = agreement.top3_shap_and_permutation & (agreement.leave_one_task_out_refit_rank <= 3)
    save('method_agreement', agreement)
    save('method_concordance', pd.DataFrame(concordance))
    cases = []
    for label, pred, category in [(0, 0, 'true_negative'), (1, 1, 'true_positive'), (0, 1, 'false_positive'), (1, 0, 'false_negative')]:
        eligible = np.flatnonzero((y == label) & (oof.prediction.to_numpy() == pred))
        if len(eligible):
            i = min(eligible, key=lambda index: ids[index])
            cases.append(dict(category=category, participant_id=ids[i], row=int(i), fold=int(folds[i]), label=label,
                prediction=pred, probability=float(oof.probability[i]), raw_score=raw[i], base_value=base[i],
                selection_rule='lexicographically first participant ID in each baseline outcome; illustrative post-hoc'))
    save('local_cases', pd.DataFrame(cases))
    safe_rows = []
    for f in range(6):
        safe = pd.read_csv(OUT / 'train_only' / f'outer_fold_{f}' / 'task_importance.csv')
        safe['outer_fold'] = f
        safe_rows.append(safe)
    pd.concat(safe_rows, ignore_index=True).to_csv(OUT / 'train_only/fold_task_importance.csv', index=False)
    write(OUT / 'validation.json', dict(complete=False, stage='tables_generated', outer_folds=6,
          baseline_oof=baseline, permutation_evaluations=len(permutation), ablation_oof_rows=len(ablation)))
    write_report(baseline, task_summary, group_summary, ablation_summary, agreement,
                 stable, permutation_summary, pd.DataFrame(concordance))
    print('TABLES GENERATED; baseline OOF:', baseline, flush=True)


def md(frame, digits=3):
    def cell(v):
        if pd.isna(v):
            return ''
        return f'{v:.{digits}f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def write_report(baseline, task, groups, ablation, agreement, stability, permutation, concordance):
    topo = task[['task', 'n_features', 'mean_absolute_shap_mass', 'share_percent', 'mean_rank', 'rank_sd', 'top3_count']]
    loto = ablation[ablation.method == 'leave_one_task_out_refit'].copy()
    for k in ('accuracy', 'f1_macro', 'auc', 'balanced_accuracy', 'delta_f1_macro', 'mean_delta_f1_macro', 'std_delta_f1_macro'):
        loto[k] *= 100
    perm = permutation[permutation.level == 'task'].copy().sort_values('drop_f1_macro_mean', ascending=False)
    for k in perm.columns:
        if k.startswith('drop_'):
            perm[k] *= 100
    baseline_row = ablation[ablation.method == 'original_baseline'].iloc[0]
    baseline_table = pd.DataFrame([dict(metric=k, OOF=100 * baseline[k],
                          fold_mean=100 * baseline_row['mean_' + k], fold_SD=100 * baseline_row['std_' + k]) for k in METRICS])
    stable_task = stability[(stability.level == 'task') & (stability.top_k == 3)]
    rho_perm = concordance[(concordance.comparison == 'SHAP_vs_permutation') & (concordance.level == 'task') & (concordance.metric == 'f1_macro')].iloc[0].spearman
    rho_loto = concordance[(concordance.comparison == 'SHAP_vs_leave_one_task_out_refit') & (concordance.metric == 'f1_macro')].iloc[0].spearman
    strong = ', '.join(agreement.loc[agreement.top3_all_three_methods, 'task'])
    outcome = pd.read_csv(TABLES / 'correct_vs_incorrect_tasks.csv')
    variability = pd.read_csv(TABLES / 'participant_rank_stability.csv')
    proven = json.loads((OUT / 'provenance.json').read_text())
    text = f'''# Kinematic Task-Level XAI — nghiên cứu trên 258 participant

Ngày: 2026-11-16 (Asia/Bangkok). HEAD đầu vào: `{proven['head']}`.
Đây là phân tích bản tái lập độc lập DiaGraMo sau refactor. Nhãn là `group`,
không đồng nhất hoàn toàn với cờ dysgraphia cụ thể trong dữ liệu. Không suy ra
biomarker, chẩn đoán lâm sàng hoặc quan hệ nhân quả từ SHAP.

## Protocol, nguồn và tính nguyên vẹn

Giữ 258 ID, 14 tác vụ × 74 đặc trưng, Stratified 6-Fold theo người, seed 42.
Assignments được đọc trực tiếp từ baseline và kiểm tra với StratifiedKFold gốc.
Sáu checkpoint LightGBM outer-fold có sẵn được dùng lại; xác suất/nhãn của chúng
phải khớp OOF cũ trước khi giải thích. Median imputer, variance filter và scaler
được kiểm tra khớp với fit chỉ trên 215 người Train tương ứng. Không dùng mô hình
fit toàn cohort cho held-out attribution. `baseline_manifest.json` bảo vệ các
artifact gốc; chỉ thư mục nghiên cứu mới được ghi output.

Mapping lấy từ `schema.json`, đối chiếu TASKS trong prepare.py và thực thi extractor
gốc trên 42 recording của ba participant đầu/giữa/cuối. Tên và giá trị phải khớp
cache. Nhóm nguồn được suy từ prefix đã kiểm chứng: có **12 nhóm** trong code
hiện tại (gồm Peaks, Lifts và Energy); không ép chúng thành số nhóm mô tả trong
bài báo. Không khôi phục module đã xóa của những nghiên cứu trước. Tái sử dụng
`evaluate.classifiers`, `build_pipeline`, `metrics`, checkpoint và fold hiện có.

Baseline kiểm chứng từ dữ liệu, đơn vị %:

{md(baseline_table)}

Dependency thực chạy: `{json.dumps(proven['versions'], ensure_ascii=False)}`.
SHAP 0.49.1 được giữ để tương thích Python 3.10 của baseline; không nâng NumPy/
LightGBM/sklearn. Seed permutation được ghi riêng cho mỗi block/fold/repeat.

## A — Feature-level TreeSHAP

TreeExplainer dùng `model_output='raw'`, `feature_perturbation='tree_path_dependent'`.
Reference là thống kê đường đi/cỡ leaf của cây đã học trên Train, không dùng
background lấy từ Test. Giải thích chỉ 43 người held-out của từng fold; ghép đúng
258 người OOF. Cột hằng bị variance filter loại nhận attribution 0 khi ánh xạ lại
1.036 cột. Scaling một-một không làm mất danh tính đặc trưng gốc.

Kiểm tra từng người: `base_value + sum(phi) = raw_score`, và
`sigmoid(raw_score) = p(group=1)`. Dung sai kiểm tra độc lập là 1e-9; sai số thực
tế ghi trong validation. SHAP ở **log-odds**, không phải xác suất; không áp dụng
sigmoid riêng cho từng phi rồi cộng. Đây là path-dependent TreeSHAP, không phải
conditional causal attribution hay một phân rã xác suất.

Top 20 đầy đủ ở `exploratory/tables/top20_features.csv`; fold-specific cho toàn
bộ 1.036 feature ở `feature_importance_by_fold.csv`. Các feature đầu bảng:

{md(pd.read_csv(TABLES / 'top20_features.csv')[['full_name','mean_absolute_shap_mass','mean_rank','rank_sd']].head(10))}

![Beeswarm](figures/shap_beeswarm.png)
![Bar plot](figures/shap_bar.png)

Bốn waterfall minh họa true-positive, true-negative, false-positive và false-negative.
Chọn ID nhỏ nhất theo thứ tự từ điển trong mỗi outcome; quy tắc được lưu ở
`local_cases.csv`, không chọn trường hợp có hình đẹp nhất. Đây là minh họa hậu
kiểm, không dùng để chọn model/threshold.

## B — Task-level attribution

Mass(task) = mean_participant(sum_feature_in_task(abs(phi))). Signed contribution
được cộng riêng trước khi xét dấu. Không thay mass bằng abs(sum(phi)): đóng góp
trái dấu có thể triệt tiêu. Mass là tổng attribution của từng feature, **không phải
grouped Shapley value** khi task được xem như một player trong một game khác.

{md(topo)}

`copy` có {float(task.set_index('task').loc['copy','share_percent']):.2f}% mass;
`dictation` và `lower-loops-fast` đứng sau. Tất cả task có cùng 74 cột nên so sánh
mass giữa task không bị khác số lượng feature. `copy` và `dictation` vào Top-3
ở cả sáu fold; vị trí thứ ba ít ổn định hơn. SD rank và SD mass qua sáu fold cùng
IQR participant được công bố, **không gọi SD là CI**. Các Train fold chồng lấn,
nên sáu fold không phải sáu cohort độc lập. Không thêm CI hoặc kiểm định giả
định độc lập cho các rank này.

![Task mass](figures/task_importance.png)

`task_signed_summary.csv` báo mean signed log-odds, mean absolute net và mass;
`participant_task_attributions.csv` lưu cả signed sum và mass cho từng người/task.
Mean signed gần 0 không có nghĩa task không quan trọng.

## C — Feature-group explanation

{md(groups[['group','n_features','share_percent','mass_per_feature','mean_rank','rank_sd']])}

FFT, Statistics và Trajectory có tổng mass lớn nhất. Sau khi chia số feature,
thứ hạng có thể thay đổi; Trajectory có mass/feature cao nhất trong lần chạy này.
Không gán ý nghĩa sinh học chỉ từ tên prefix hoặc mean absolute importance.

![Groups](figures/feature_groups.png)
![Task × Group](figures/task_group_heatmap.png)

## D — Grouped permutation trên held-out folds

Trong mỗi Test fold, toàn bộ cột của một task hoặc source group được hoán vị
bằng **cùng một thứ tự participant**; các cột khác và nhãn giữ nguyên. Mỗi block
lặp 30 lần, seed và thứ tự hoán vị được lưu cùng dự đoán. Tổng 26 block × 30 × 6
= 4.680 lượt. Không fit lại classifier. Drop = baseline − permuted, dương là giảm
hiệu năng sau perturbation. Bảng dưới là mean ± SD giữa **sáu fold mean**, sau khi
trung bình 30 repeats trong mỗi fold; 180 phép lặp không phải 180 cohort độc lập.

{md(perm[['block','drop_f1_macro_mean','drop_f1_macro_std','drop_accuracy_mean','drop_auc_mean','drop_balanced_accuracy_mean']])}

Permutation giữ cấu trúc nội bộ block nhưng có thể tạo tổ hợp **out-of-distribution**
giữa task/group đã hoán vị và các task/group khác. Drop phản ánh reliance của
mô hình đã fit, không phải causal effect hoặc giá trị khi được huấn luyện lại.

![Permutation tasks](figures/permutation_task.png)
![Permutation groups](figures/permutation_group.png)

## E — Task ablation: hai câu hỏi khác nhau

1. **Test-time removal diagnostic:** thay 74 cột raw của task bằng median chỉ từ
Train fold (statistics của imputer đã fit), giữ nguyên mô hình và 1.036 cột.
Đây là một quy tắc operational cụ thể cho missing task, có thể OOD; không coi
việc ghi số 0 vào cột raw là trung tính. Bảng kết quả đầy đủ có prefix method
`test_time_train_median`.
2. **Leave-One-Task-Out retraining:** bỏ đúng 74 cột ở cả Train và Test rồi fit
lại đủ preprocessing + LightGBM với cùng tham số, seed và assignments. Có 14 × 6
= **84 fit** trên 962 cột. Không chọn task subset/hyperparameter hoặc ngưỡng từ
outer Test. Checkpoint mới nằm riêng, baseline không bị ghi đè.

Bảng LOTO dưới dùng OOF gộp (%); delta F1 = ablated − baseline, đơn vị điểm %.
`corrected/harmed` là số người chuyển từ sai→đúng/đúng→sai so với baseline.

{md(loto[['task','accuracy','f1_macro','auc','balanced_accuracy','delta_f1_macro','corrected','harmed']])}

CSV đầy đủ còn báo mean±SD mỗi fold, **paired fold deltas**, cả bốn metric,
7.224 dòng thay đổi dự đoán theo participant cho hai phương pháp × 14 task.
Không đánh đồng hai thí nghiệm. Mức tụt OOF đơn lẻ không phải kiểm định ý nghĩa
thống kê hoặc bằng chứng số task tối thiểu.

![Task ablation](figures/task_ablation.png)

## F — Stability, agreement và disagreement

Rank correlation task giữa 15 cặp fold có mean Spearman
**{stable_task.spearman.mean():.3f}** (min {stable_task.spearman.min():.3f},
max {stable_task.spearman.max():.3f}); Top-3 Jaccard mean
**{stable_task.jaccard.mean():.3f}**. Feature-level rank yếu ổn định hơn task/group;
toàn bộ kết quả ở `fold_stability.csv`, không chỉ trình bày cặp fold thuận lợi.

![Ranks](figures/task_fold_ranks.png)
![Across methods](figures/method_rank_agreement.png)

Spearman SHAP task mass ↔ permutation F1 drop là **{rho_perm:.3f}**, nhưng với
LOTO retraining F1 drop chỉ **{rho_loto:.3f}**. Top-3 đồng thời ở SHAP, permutation
và LOTO: **{strong}**. `big-spiral-fast` đứng thứ 9 theo SHAP nhưng thứ 2 theo
LOTO; `saw-fast` thứ 13 theo SHAP nhưng nằm khoảng thứ 4–5 theo LOTO. Ngược lại,
`dictation` rất lớn theo SHAP/permutation nhưng đứng thứ 7 theo LOTO. Khi bỏ cột
và fit lại, cây có thể đổi cấu trúc và phân phối đóng góp giữa các feature tương
quan; redundancy và tính không ổn định khi refit là các khả năng cần kiểm chứng,
không phải nguyên nhân nhân quả đã được chứng minh.

Không lựa chọn phương pháp XAI thuận lợi nhất. Bảng agreement công bố cả rank
và drop, kể cả khi chúng bất đồng. Tất cả 14 LOTO đều có Macro-F1 OOF thấp hơn
baseline trong lần chạy này; điều đó chưa đủ khẳng định mỗi task là không thể
thay thế hoặc nên giữ mọi task trong mọi mô hình khác.

Có {int(outcome[outcome.correct].n.iloc[0])} dự đoán đúng và
{int(outcome[~outcome.correct].n.iloc[0])} sai. Importance và rank của từng người
được lưu đầy đủ; median Spearman rank cá nhân với thứ hạng task toàn cohort là
{variability.spearman_to_global.median():.3f}. Thống kê này mang tính mô tả và tham
chiếu toàn cohort có bao gồm chính người đó; không phải một phép đo validation
độc lập. Nhóm đúng/sai có cỡ mẫu khác nhau và cùng chia sẻ model trong fold;
không dùng sự khác nhau để kết luận về bệnh lý.

![Participant variability](figures/participant_rank_variability.png)

## Quy trình train-only cho nghiên cứu tiếp theo

`diagramo.kinematic_xai.train_only_task_importance` chỉ nhận X/y/ID của **Train
đang xét**, tự cross-fit 3-fold bên trong nó, rồi tính importance từ các hàng
inner held-out. Mỗi participant được giải thích đúng một lần bởi model chưa
thấy người đó. `forbidden_ids` chặn ID ngoài Train. Mỗi outer Train có một bộ
artifact demo riêng tại `train_only/outer_fold_f/`; tổng **18 fit**, tách khỏi
`exploratory/`. Không chọn subset hoặc trọng số task trong nhiệm vụ này.

Khi tích hợp vào inner CV của một mô hình tương lai, phải gọi lại API trên
**inner Train**, với inner Validation và outer Test nằm trong forbidden_ids:

```python
importance, phi, fold_ids = train_only_task_importance(
    X[inner_train], y[inner_train], ids[inner_train], mapping, factory,
    seed=42, forbidden_ids=np.r_[ids[inner_valid], ids[outer_test]])
```

**Không** tải importance đã tính từ toàn outer Train để chọn task/weight cho
inner Validation nằm bên trong chính outer Train đó. Artifact demo outer Train
chỉ phù hợp với quyết định fit trên toàn outer Train tương ứng; cấp inner CV
cần importance được tính lại từ cấp inner Train.

Outer-test XAI hiện tại là phân tích khoa học thăm dò. Nếu chọn task, weight hoặc
kiến trúc theo các bảng này rồi dùng cùng outer Test để tuyên bố một mô hình mới
tốt hơn một cách không thiên lệch thì đã có feedback từ dữ liệu đánh giá. Một
quy trình train-only đúng ở cấp code không xóa được việc nhà nghiên cứu đã xem
kết quả toàn cohort. Cần mô tả tính thăm dò và ưu tiên cohort/hold-out mới để xác
nhận thiết kế tương lai; không coi những fold đã xem là một phép đánh giá mới độc lập.

## Giả thuyết cho Motor-Invariant Task Transformer ở bước sau

- Xét `copy` và `lower-loops-fast` là ứng viên thông tin mạnh, nhưng không gán
trọng số cố định từ outer-test XAI hiện tại.
- Kiểm tra khả năng học đặc trưng motor dùng chung qua source groups, đồng thời
giữ task context: dominance của copy/dictation có thể bao gồm đặc tính riêng tác
vụ; đây là giả thuyết, chưa xác lập representation motor-invariant.
- Những cặp task fast/precise có cùng dạng tên trong schema là ứng viên để kiểm
tra invariance và phụ thuộc tốc độ, với đối chiếu riêng trên Train; không mặc
định chúng đo cùng một cơ chế sinh học.
- Bất đồng SHAP–LOTO cho thấy cần đánh giá task redundancy/interactions và độ ổn
định khi refit, thay vì loại task chỉ theo mass nhỏ. Task subset phải được chọn
trong Train với nested protocol; chưa biết số task tối thiểu.

Nhiệm vụ này không triển khai Transformer, Knowledge Graph hoặc thí nghiệm trên
nhánh embedding. Không có lựa chọn mô hình mới dựa trên các metric ở đây.

## Artifact, chạy lại và validation

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/analyze_kinematic_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/report_kinematic_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/plot_kinematic_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/validate_kinematic_xai.py --numerics-only
.venv/Scripts/python.exe -X utf8 -B scripts/create_kinematic_xai_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 05_kinematic_task_xai.ipynb --validation-output results/kinematic_xai/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_kinematic_xai.py
```

Checkpoint LOTO và train-only lưu cục bộ ở `checkpoints/` và Git ignore; có thể
tạo lại bằng script với input/code signature đã ghi. Bảng số và thứ tự permutation
được lưu cùng OOF để replay. Thay code phân tích cần output mới, không trộn lượt
chạy khác. Notebook là viewer đọc artifact, bảng số trước hình, không fit model.
Chỉ xem nghiên cứu hoàn tất khi `validation.json` có `complete=true`, mọi fold,
permutation, ablation, train-only provenance, hình, notebook và kiểm thử đã đạt.

## Tài liệu/phần mềm tham khảo

- [SHAP TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
- Quy trình tái lập: `../../EXPERIMENT_PROTOCOL.md`.
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
  *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
  [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
  Tham khảo hướng dẫn SHAP và scientific-visualization; metadata arXiv được
  kiểm tra ngày 2026-11-16, bản hiện hành v2.
'''
    (OUT / 'REPORT_VI.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
