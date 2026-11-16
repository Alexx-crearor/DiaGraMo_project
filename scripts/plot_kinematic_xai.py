"""Publication-oriented, descriptive PNG/SVG figures from verified XAI tables."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shap
from kinematic_xai_common import ROOT, OUT, sha, write
from report_kinematic_xai import TABLES, load_shap
from diagramo.kinematic_xai import aggregate

FIGURES = OUT / 'figures'


def main():
    np.random.seed(42)
    FIGURES.mkdir(parents=True, exist_ok=True)
    X, y, ids, folds, oof, phi, base, raw = load_shap()
    mapping = pd.read_csv(OUT / 'feature_mapping.csv')
    task = pd.read_csv(TABLES / 'task_importance.csv')
    group = pd.read_csv(TABLES / 'group_importance.csv')
    tasks = task.task.tolist()
    feature_names = mapping.full_name.tolist()
    manifest = []
    def save(fig, name, description, sources, n=258, uncertainty='descriptive only; no confidence interval'):
        fig.savefig(FIGURES / (name + '.png'), dpi=300, facecolor='white')
        fig.savefig(FIGURES / (name + '.svg'), dpi=300, facecolor='white', metadata={'Date': None})
        paths = [ROOT / p if not str(p).startswith(str(ROOT)) else p for p in sources]
        manifest.append(dict(name=name, description=description, n=n, unit='participant',
             dpi=300, plot_seed=42, width_inches=float(fig.get_figwidth()), height_inches=float(fig.get_figheight()),
             uncertainty=uncertainty, publisher_profile='general research draft; target journal unspecified',
             sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
             outputs={ext: sha(FIGURES / (name + '.' + ext)) for ext in ('png', 'svg')}))
        plt.close(fig)
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'kinematic_xai_seed42', 'figure.facecolor': 'white'}):
        explanation = shap.Explanation(values=phi, base_values=base, data=X, feature_names=feature_names)
        columns = np.argsort(-np.abs(phi).mean(axis=0), kind='stable')[:20]
        shap.plots.beeswarm(explanation[:, columns], max_display=20, show=False, group_remaining_features=False)
        fig = plt.gcf()
        fig.set_size_inches(13, 9)
        fig.subplots_adjust(left=.43, right=.94, top=.92, bottom=.08)
        fig.axes[0].set(xlabel='TreeSHAP value (raw log-odds)', title='Top 20 features: held-out participants, n=258')
        save(fig, 'shap_beeswarm', 'Top 20 individual features by mean absolute raw TreeSHAP; color is original feature value within each feature.',
             [TABLES / 'top20_features.csv', OUT / 'feature_mapping.csv',
              *[OUT / 'exploratory/shap' / f'fold_{f}.npz' for f in range(6)]])
        shap.plots.bar(explanation[:, columns], max_display=20, show=False)
        fig = plt.gcf()
        fig.set_size_inches(13, 9)
        fig.subplots_adjust(left=.43, right=.95, top=.92, bottom=.08)
        fig.axes[0].set(xlabel='Mean |TreeSHAP| (raw log-odds)', title='Top 20 individual features; n=258')
        save(fig, 'shap_bar', 'Twenty individual features only; no summed remainder bar.', [TABLES / 'top20_features.csv'])
        fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
        rev = task.iloc[::-1]
        ax.barh(rev.task, rev.mean_absolute_shap_mass, xerr=rev.fold_mass_sd, capsize=3, color='#0072B2')
        ax.set(xlim=(0, None), xlabel='Mean sum |TreeSHAP| within task (log-odds mass)',
               title='Task attribution mass: mean and SD across six outer folds')
        save(fig, 'task_importance', 'Task attribution mass includes 74 features per task; not a causal effect.',
             [TABLES / 'task_importance.csv'], uncertainty='sample SD across six outer folds; overlapping training sets, not CI')
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
        g = group.iloc[::-1]
        axes[0].barh(g.group, g.share_percent, color='#0072B2')
        axes[0].set(xlim=(0, None), xlabel='Share of total |TreeSHAP| mass (%)', title='Total group contribution')
        axes[1].barh(g.group, g.mass_per_feature, color='#D55E00')
        axes[1].set(xlim=(0, None), xlabel='Mean |TreeSHAP| per feature (log-odds)', title='Adjusted for number of features')
        save(fig, 'feature_groups', 'Twelve groups are inferred from actual source feature prefixes; group sizes differ.', [TABLES / 'group_importance.csv'])
        matrix = pd.read_csv(TABLES / 'task_group_matrix.csv').pivot(index='task', columns='group', values='mean_absolute_shap_mass').loc[tasks, group.group]
        fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
        im = ax.imshow(matrix, cmap='viridis', vmin=0, aspect='auto', interpolation='nearest')
        ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns, yticks=range(14), yticklabels=tasks,
               title='Task × source feature group: mean held-out attribution mass')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.colorbar(im, ax=ax, label='Mean sum |TreeSHAP| (log-odds mass)')
        save(fig, 'task_group_heatmap', 'One shared linear color scale across all task/group cells; no row normalization.', [TABLES / 'task_group_matrix.csv'])
        ranks = pd.read_csv(TABLES / 'task_importance_by_fold.csv').pivot(index='task', columns='fold', values='rank').loc[tasks]
        fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
        im = ax.imshow(ranks, cmap='viridis_r', vmin=1, vmax=14, aspect='auto', interpolation='nearest')
        ax.set(xticks=range(6), xticklabels=['Fold ' + str(i) for i in range(6)], yticks=range(14), yticklabels=tasks,
               title='Task importance ranks across outer folds')
        for i in range(14):
            for j in range(6):
                value = ranks.iloc[i, j]
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', color='white' if value > 7 else 'black')
        fig.colorbar(im, ax=ax, label='Rank (1 = most attribution mass)')
        save(fig, 'task_fold_ranks', 'All six task rankings; each outer-test fold contains 43 participants.', [TABLES / 'task_importance_by_fold.csv'])
        perm = pd.read_csv(TABLES / 'permutation_importance.csv')
        for level in ('task', 'group'):
            data = perm[perm.level == level].sort_values('drop_f1_macro_mean')
            fig, axes = plt.subplots(1, 4, figsize=(17, 7), layout='constrained', sharey=True)
            for ax, metric, label in zip(axes, ('f1_macro', 'accuracy', 'auc', 'balanced_accuracy'), ('Macro-F1', 'Accuracy', 'ROC-AUC', 'Balanced accuracy')):
                ax.errorbar(data['drop_' + metric + '_mean'] * 100, np.arange(len(data)),
                    xerr=data['drop_' + metric + '_std'] * 100, fmt='o', color='#0072B2', capsize=3)
                ax.axvline(0, color='#555555', linestyle='--', linewidth=.8)
                ax.set(yticks=range(len(data)), yticklabels=data.block, xlabel=label + ' drop (pp)', title=label)
            fig.suptitle(f'Grouped {level} permutation: 30 repeats per outer fold; n=258')
            save(fig, 'permutation_' + level, 'Performance drops are original minus permuted. Dots average 30 repeats within each fold, then six folds.',
                 [TABLES / 'permutation_importance.csv'], uncertainty='sample SD across six fold means; not across 180 independent participants or a CI')
        ablation = pd.read_csv(TABLES / 'task_ablation_summary.csv')
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout='constrained', sharey=True)
        ordering = tasks[::-1]
        for ax, metric, label in zip(axes, ('f1_macro', 'accuracy'), ('Macro-F1', 'Accuracy')):
            for offset, method, color, marker, text in [(-.15, 'test_time_train_median', '#0072B2', 'o', 'Test-time train-median replacement'),
                                                       (.15, 'leave_one_task_out_refit', '#D55E00', 's', 'Leave-one-task-out retraining')]:
                frame = ablation[ablation.method == method].set_index('task').loc[ordering]
                ax.errorbar(-100 * frame['mean_delta_' + metric], np.arange(14) + offset,
                            xerr=100 * frame['std_delta_' + metric], fmt=marker, capsize=2, color=color, label=text)
            ax.axvline(0, color='#555555', linestyle='--', linewidth=.8)
            ax.set(yticks=range(14), yticklabels=ordering, xlabel='Paired fold ' + label + ' drop (pp)', title=label)
        axes[1].legend(fontsize=8, loc='lower right')
        fig.suptitle('Task ablation: model reliance and retraining answer different questions')
        save(fig, 'task_ablation', 'Paired fold performance drops relative to the exact original Kinematic baseline; all 14 omissions shown.',
             [TABLES / 'task_ablation_summary.csv', TABLES / 'task_ablation_by_fold.csv'], uncertainty='sample SD of six paired fold deltas; not CI')
        agreement = pd.read_csv(TABLES / 'method_agreement.csv').set_index('task').loc[tasks]
        rank_columns = ['shap_rank', 'permutation_rank', 'test_time_train_median_rank', 'leave_one_task_out_refit_rank']
        fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
        im = ax.imshow(agreement[rank_columns], cmap='viridis_r', vmin=1, vmax=14, aspect='auto', interpolation='nearest')
        ax.set(xticks=range(4), xticklabels=['SHAP mass', 'Permutation', 'Median replacement', 'LOTO retraining'],
               yticks=range(14), yticklabels=tasks, title='Task ranks: agreement and disagreement across explanation methods')
        for i in range(14):
            for j in range(4):
                value = agreement.iloc[i][rank_columns[j]]
                ax.text(j, i, f'{value:g}', ha='center', va='center', color='white' if value > 7 else 'black')
        fig.colorbar(im, ax=ax, label='Rank; performance methods ranked by Macro-F1 drop')
        save(fig, 'method_rank_agreement', 'SHAP uses attribution mass; permutation uses mean-fold F1 drop; median and retraining use OOF F1 drop. Ties use average ranks.', [TABLES / 'method_agreement.csv'])
        participants = pd.read_csv(TABLES / 'participant_task_attributions.csv')
        fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
        for offset, correct, color in [(-.18, True, '#0072B2'), (.18, False, '#D55E00')]:
            data = [participants[(participants.task == t) & (participants.correct == correct)]['rank'] for t in ordering]
            b = ax.boxplot(data, positions=np.arange(14) + offset, vert=False, widths=.28, patch_artist=True,
                           manage_ticks=False, showfliers=True, medianprops=dict(color='black'),
                           flierprops=dict(marker='.', markersize=2, markeredgecolor=color))
            for box in b['boxes']:
                box.set(facecolor=color, alpha=.7)
        ax.set(yticks=range(14), yticklabels=ordering, xlabel='Participant-specific task rank', xlim=(.5, 14.5),
               title='Task-rank variability among correct and incorrect baseline predictions')
        ax.legend(handles=[Line2D([0], [0], color='#0072B2', linewidth=6, label=f'Correct (n={int((oof.prediction.to_numpy()==y).sum())})'),
                           Line2D([0], [0], color='#D55E00', linewidth=6, label=f'Incorrect (n={int((oof.prediction.to_numpy()!=y).sum())})')],
                  loc='upper left', bbox_to_anchor=(0, -.10), ncol=2)
        save(fig, 'participant_rank_variability', 'Boxes show participant medians/IQR and Tukey whiskers/outliers, not uncertainty of estimated means. Unequal outcome group sizes.',
             [TABLES / 'participant_task_attributions.csv'], uncertainty='empirical participant distributions; descriptive, not CI')
        cases = pd.read_csv(TABLES / 'local_cases.csv')
        for case in cases.itertuples():
            shap.plots.waterfall(explanation[int(case.row)], max_display=12, show=False)
            fig = plt.gcf()
            fig.set_size_inches(14, 8)
            fig.subplots_adjust(left=.44, right=.82, top=.88, bottom=.1)
            fig.axes[0].set_title(f'{case.category}: {case.participant_id}, outer fold {case.fold}\np(group=1)={case.probability:.3f}; attribution units: raw log-odds', pad=35)
            save(fig, 'waterfall_' + case.category, 'Lexicographically first participant in each outcome category; illustrative post-hoc selection. The summed remainder preserves raw-score additivity.',
                 [TABLES / 'local_cases.csv', EXP_PATH(case.fold)], n=1)
        for case in cases.itertuples():
            signed = aggregate(phi, mapping, 'task', absolute=False).to_numpy()[int(case.row)]
            task_explanation = shap.Explanation(values=signed, base_values=base[int(case.row)],
                data=None, feature_names=list(aggregate(phi, mapping, 'task').columns))
            if case.category == 'false_negative':
                shap.plots.waterfall(task_explanation, max_display=14, show=False)
                fig = plt.gcf()
                fig.set_size_inches(12, 8)
                fig.subplots_adjust(left=.35, right=.82, top=.88, bottom=.1)
                fig.axes[0].set_title(f'Signed sums by task: {case.participant_id}\nSums of feature SHAP, not grouped Shapley values', pad=35)
                save(fig, 'task_signed_waterfall', 'Signed feature contributions summed within tasks; retains cancellation and exact raw-score sum.',
                     [TABLES / 'participant_task_attributions.csv', TABLES / 'local_cases.csv'], n=1)
        write(FIGURES / 'manifest.json', manifest)
    print('FIGURES', len(manifest), 'PNG + SVG, 300 DPI', flush=True)


def EXP_PATH(fold):
    return OUT / 'exploratory/shap' / f'fold_{fold}.npz'


if __name__ == '__main__':
    main()
