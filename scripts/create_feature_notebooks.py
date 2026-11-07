"""Create two read-only experiment viewers; no feature fitting or training is run."""
from pathlib import Path
from textwrap import dedent
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def md(source):
    return nb.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    return nb.v4.new_code_cell(dedent(source).strip())


SETUP=r'''
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.6f}'.format)

ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
OUT = ROOT / 'results/feature_optimization'
validation = json.loads((OUT / 'validation.json').read_text(encoding='utf-8'))
assert validation['complete'], 'Hoàn tất và validate thực nghiệm trước khi xem.'
summary = pd.read_csv(OUT / 'summary.csv')
BRANCHES = ['Kinematic', 'MOMENT', 'GAF', 'CWT']
METRICS = [('accuracy', 'Accuracy'), ('f1_macro', 'F1 macro'), ('auc', 'ROC-AUC')]
COLORS = {'none':'#555555', 'pca':'#0072B2', 'anova':'#D55E00', 'extratrees':'#009E73'}
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
exports = []
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'svg.fonttype':'none'})

def show(fig, name, description, sources):
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=180, facecolor='white')
    exports.append(dict(name=name, description=description, sources=sources,
                        formats=['png','svg'], n_participants=258,
                        uncertainty='Fold SD where shown; not a confidence interval'))
    plt.show()
    plt.close(fig)

def save_manifest(name):
    for item in exports:
        item['source_sha256']={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest()
                               for p in item['sources']}
    (FIG / name).write_text(json.dumps(exports, ensure_ascii=False, indent=2), encoding='utf-8')

print('Project:', ROOT)
print('Validated configurations:', len(summary), '| Participants: 258 | Outer folds: 6 | Seed: 42')
'''


def main():
    overview=[
        md('''
        # 05. Feature selection và dimensionality reduction

        Xem kết quả đã chạy của **Kinematic, MOMENT, GAF/ViT-B16, CWT/EfficientNet-B4**.
        Notebook chỉ đọc artifact; không fit lại PCA, selector hoặc classifier.
        216 cấu hình gồm 12 baseline OOF cũ và 204 cấu hình giảm chiều.

        Các bảng xếp hạng là **thăm dò trên cùng 6 fold**, có thiên lệch chọn cấu hình.
        Xem notebook **06** để đánh giá quy trình chọn bằng nested CV.
        Bảng số đầu notebook giữ metric ở thang 0–1 như CSV gốc; biểu đồ dùng %.
        Uncertainty là SD giữa 6 fold, không phải CI.
        '''), code(SETUP),
        md('''
        ## Kết quả cuối cùng dạng số — Top 20 cấu hình

        Bảng DataFrame thuần số như notebook đánh giá cũ. `accuracy`, `f1_macro`, `auc`,
        v.v. là metric tính gộp từ OOF; `mean_*` và `std_*` là trung bình và SD giữa 6 fold.
        Tất cả giữ thang **0–1**, không nhân 100. `mean_features` là số chiều thực tế
        trung bình sau tiền xử lý/giảm chiều; `raw_features` là chiều đầu vào gốc.

        Xếp theo **mean F1 macro**, rồi mean Accuracy và mean AUC, đúng tiêu chí thực nghiệm.
        Đây là bảng thăm dò; bảng kết luận về quy trình chọn nằm đầu notebook 06.
        '''),
        code('''
        result_columns = ['representation','reducer','classifier','raw_features','mean_features',
                          'accuracy','f1_macro','f1_weighted','f1_binary','auc','balanced_accuracy',
                          'mean_accuracy','mean_f1_macro','mean_auc',
                          'std_accuracy','std_f1_macro','std_auc']
        final_results = summary.sort_values(
            ['mean_f1_macro','mean_accuracy','mean_auc'], ascending=False
        )[result_columns].reset_index(drop=True)
        final_results.head(20)
        '''),
        md('''
        ## Kết quả tốt nhất từng nhánh — bao gồm lựa chọn giữ baseline

        Chọn trong các cấu hình đã khảo sát, gồm `none`. Vì vậy Kinematic có thể giữ
        baseline đầy đủ chiều nếu giảm chiều không tốt hơn. Vẫn là xếp hạng thăm dò.
        '''),
        code('''
        best_final = summary.sort_values(
            ['mean_f1_macro','mean_accuracy','mean_auc'], ascending=False
        ).groupby('branch', sort=False).head(1)
        best_final.set_index('branch').loc[BRANCHES].reset_index()[['branch']+result_columns]
        '''),
        md('''
        ## Toàn bộ 216 cấu hình — đầy đủ bảng số

        Không cắt hàng hoặc cột; có thể dùng tìm kiếm trong output để tìm reducer/classifier.
        Để lọc thủ công, ví dụ: `final_results.query("representation == 'moment'")`.
        '''),
        code('''
        with pd.option_context('display.max_rows', None):
            display(final_results)
        '''),
        md('''
        ## Bảng so sánh với baseline cũ tốt nhất

        Xếp hạng bằng mean F1 macro, rồi Accuracy và AUC. Baseline tốt nhất cũ có thể
        dùng classifier khác; bảng cùng classifier ở cell dưới giúp tách hai ảnh hưởng.
        '''),
        code('''
        best = pd.read_csv(OUT / 'best_per_branch.csv').set_index('branch').loc[BRANCHES].reset_index()
        cols = ['branch','reducer','classifier','raw_features','mean_features','old_best_classifier']
        table = best[cols].copy()
        for m, label in METRICS:
            table['Old '+label+' (%)'] = 100*best['old_best_mean_'+m]
            table['New '+label+' (%)'] = 100*best['mean_'+m]
        display(table.round(2))
        same = pd.read_csv(OUT / 'comparison_same_classifier.csv')
        display(best[['branch','reducer','classifier','delta_f1_macro_pp',
                      'delta_vs_old_best_f1_macro_pp']].round(2))
        '''),
        md('''
        ## Chênh lệch so với baseline cùng classifier — bảng số

        Chỉ so sánh các cấu hình dùng cùng classifier. Metric giữ thang 0–1;
        các cột `delta_*_pp` là chênh lệch **điểm phần trăm**. Bảng này gồm
        ứng viên giảm chiều có F1 cao nhất cho từng cặp nhánh/classifier.
        '''),
        code('''
        same_best = same.sort_values(
            ['mean_f1_macro','mean_accuracy','mean_auc'], ascending=False
        ).groupby(['branch','classifier'], sort=False).head(1)
        same_columns = ['branch','reducer','classifier','mean_features']
        for m, _ in METRICS:
            same_columns += ['baseline_mean_'+m, 'mean_'+m, 'delta_'+m+'_pp']
        same_best[same_columns].sort_values(['branch','classifier']).reset_index(drop=True)
        '''),
        md('''
        ## Các phương pháp trên cùng một thang đo

        Mỗi cột là cấu hình có mean F1 macro cao nhất trong một phương pháp/nhánh,
        bao gồm baseline trong ba classifier đang khảo sát. Thanh sai số là SD của
        metric tương ứng. Chọn theo F1 nên cột Accuracy/AUC không phải cực đại riêng.
        '''),
        code('''
        ranked = summary.sort_values(['mean_f1_macro','mean_accuracy','mean_auc'], ascending=False)
        winners = ranked.groupby(['branch','method'], sort=False).head(1)
        methods = ['none','pca','anova','extratrees']
        fig, axes = plt.subplots(1,3,figsize=(17,5),layout='constrained')
        x = np.arange(4); width = .19
        for ax,(m,label) in zip(axes,METRICS):
            for j, method in enumerate(methods):
                g = winners[winners.method==method].set_index('branch').loc[BRANCHES]
                ax.bar(x+(j-1.5)*width,100*g['mean_'+m],width,
                       yerr=100*g['std_'+m],capsize=2,label=method,
                       color=COLORS[method],hatch=['','//','..','xx'][j])
            ax.set(xticks=x,xticklabels=BRANCHES,ylim=(0,100),ylabel=label+' (%)')
            ax.grid(axis='y',alpha=.2)
        axes[0].legend(title='Method',fontsize=8)
        fig.suptitle('Best macro-F1 candidate per method | mean and SD of 6 folds | exploratory')
        show(fig,'method_comparison','Method winners chosen by macro-F1; bars show each metric and fold SD.', ['summary.csv'])
        display(winners[['branch','method','reducer','classifier','mean_features',
                         'mean_accuracy','mean_f1_macro','mean_auc']].round(4))
        '''),
        md('''
        ## Mức giảm chiều và hiệu năng — hiển thị mọi cấu hình

        Mỗi hàng là reducer, mỗi cột là classifier; không chỉ hiển thị cấu hình thắng.
        Các panel dùng chung thang màu F1 macro 0–100%. Tên PCA `var90/95/99` là tỷ lệ
        phương sai train, số thành phần thực tế thay đổi theo fold.
        '''),
        code('''
        reducers = ['none','pca_16','pca_32','pca_64','pca_128','pca_var90','pca_var95','pca_var99',
                    'anova_32','anova_64','anova_128','anova_256','anova_512',
                    'extratrees_32','extratrees_64','extratrees_128','extratrees_256','extratrees_512']
        classifiers = ['LogisticRegression','SVM','LightGBM']
        fig, axes = plt.subplots(1,4,figsize=(16,10),layout='constrained')
        for ax,branch in zip(axes,BRANCHES):
            grid = summary[summary.branch==branch].pivot(index='reducer',columns='classifier',values='mean_f1_macro')
            grid = 100*grid.reindex(index=reducers,columns=classifiers)
            assert not grid.isna().any().any()
            im = ax.imshow(grid.to_numpy(),vmin=0,vmax=100,cmap='cividis',aspect='auto')
            for i in range(len(reducers)):
                for j in range(3):
                    v = grid.iloc[i,j]
                    ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8,
                            color='black' if v>55 else 'white')
            ax.set(title=branch,xticks=range(3),xticklabels=['LR','SVM','LGBM'],
                   yticks=range(len(reducers)),yticklabels=reducers)
        fig.colorbar(im,ax=axes,label='Mean F1 macro (%)',shrink=.7)
        show(fig,'all_configurations','All 216 candidates; shared F1 color limits, annotated values.', ['summary.csv'])
        '''),
        md('''
        ## Đường cong theo số chiều

        Giữ classifier riêng trên từng panel. PCA phương sai được hiển thị ở bảng,
        không trộn vào đường PCA số chiều cố định. Đường ngang là baseline cùng classifier.
        Trục x log₁₀; trục y dùng toàn khoảng 0–100%.
        '''),
        code('''
        fig, axes = plt.subplots(4,3,figsize=(16,13),layout='constrained')
        for i,branch in enumerate(BRANCHES):
            for j,clf in enumerate(classifiers):
                ax = axes[i,j]
                g = summary[(summary.branch==branch)&(summary.classifier==clf)]
                for method,marker in [('pca','o'),('anova','s'),('extratrees','^')]:
                    q = g[(g.method==method)&~g.reducer.str.contains('var')].sort_values('mean_features')
                    ax.plot(q.mean_features,100*q.mean_f1_macro,marker=marker,
                            color=COLORS[method],label=method)
                base = g[g.method=='none'].iloc[0]
                ax.axhline(100*base.mean_f1_macro,color=COLORS['none'],ls='--',label='baseline')
                ax.set(xscale='log',ylim=(0,100),title=branch+' / '+clf,
                       xlabel='Retained dimensions (log scale)',ylabel='Mean F1 macro (%)')
                ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        show(fig,'dimension_curves','Fixed dimension grids by branch and classifier; dashed same-classifier baseline.', ['summary.csv'])
        display(summary[summary.reducer.str.contains('pca_var')][
            ['branch','reducer','classifier','min_features','max_features','mean_explained_variance','mean_f1_macro']].round(4))
        '''),
        md('''
        ## Mức nén của các ứng viên đứng đầu

        Đây là giảm kích thước đầu vào classifier, chưa phải đo tốc độ toàn pipeline.
        Encoder vẫn phải tạo embedding gốc. Chưa có cơ sở thay baseline chỉ từ bảng thắng.
        '''),
        code('''
        fig, ax = plt.subplots(figsize=(10,4),layout='constrained')
        values = best.reduction_pct.to_numpy()
        ax.barh(BRANCHES,values,color=[COLORS[m] for m in best.method])
        for i,r in enumerate(best.itertuples()):
            ax.text(2,i,f'{r.raw_features:,} -> {r.mean_features:.1f} | {r.reduction_pct:.2f}%',
                    va='center',color='white',fontweight='bold')
        ax.set(xlim=(0,100),xlabel='Removed dimensions (%)',title='Compression of exploratory winners')
        show(fig,'compression','Raw versus retained feature counts for the best reduced candidate per branch.', ['best_per_branch.csv'])
        save_manifest('overview_manifest.json')
        print('Figures:', FIG)
        '''),
    ]
    nested=[
        md('''
        # 06. Nested CV, ROC và độ ổn định đặc trưng

        Outer **Stratified 6-Fold, seed 42, theo participant** giữ nguyên baseline.
        Trong mỗi outer train, inner 3-fold chọn reducer, mức chiều và classifier.
        Đối chứng `baseline` cũng chọn classifier trong inner CV nhưng không giảm chiều.

        Notebook này đánh giá **quy trình chọn cấu hình**, không xác nhận riêng cấu hình
        thắng bảng 05. Không huấn luyện lại hoặc chọn threshold từ dự đoán test.
        '''), code(SETUP),
        md('''
        ## Kết quả cuối cùng dạng số — nested CV

        Ưu tiên bảng này khi kết luận về **quy trình tối ưu**. Metric giữ thang 0–1,
        tách rõ OOF gộp, trung bình và SD qua 6 outer fold. `baseline` và `optimized`
        đều chọn cấu hình bằng inner CV, không dùng outer test để chọn.
        '''),
        code('''
        nested_final = pd.read_csv(OUT / 'nested_summary.csv')
        numeric_columns = ['branch','scenario','raw_features','mean_features',
                           'accuracy','f1_macro','f1_weighted','f1_binary','auc','balanced_accuracy',
                           'mean_accuracy','mean_f1_macro','mean_auc',
                           'std_accuracy','std_f1_macro','std_auc']
        nested_final[numeric_columns].sort_values(['branch','scenario']).reset_index(drop=True)
        '''),
        md('''
        ## Baseline → tối ưu và chênh lệch

        Cột `baseline_*` và `optimized_*` là mean metric qua 6 fold, thang 0–1.
        Cột `delta_*_pp` là chênh lệch điểm phần trăm.
        '''),
        code('''
        nested_comparison_numbers = pd.read_csv(OUT / 'nested_comparison.csv')
        nested_comparison_numbers.set_index('branch').loc[BRANCHES].reset_index()
        '''),
        code('''
        nested = pd.read_csv(OUT / 'nested_summary.csv')
        compare = pd.read_csv(OUT / 'nested_comparison.csv').set_index('branch').loc[BRANCHES]
        choices = pd.read_csv(OUT / 'selected_by_fold.csv')
        oof = pd.read_csv(OUT / 'nested_oof.csv')
        stability = pd.read_csv(OUT / 'feature_stability.csv')
        display(compare.round(4))
        '''),
        md('''
        ## Quy trình tối ưu so với quy trình baseline

        Mean ± SD của 6 outer fold; cả hai quy trình dùng cùng participant test.
        SD không phải CI và các tập train chồng lấp, không dùng kiểm định độc lập trên 6 fold.
        '''),
        code('''
        fig, axes = plt.subplots(1,3,figsize=(16,5),layout='constrained')
        x = np.arange(4)
        for ax,(m,label) in zip(axes,METRICS):
            for offset,scenario,color,hatch in [(-.18,'baseline','#555555',''),(.18,'optimized','#0072B2','//')]:
                g = nested[nested.scenario==scenario].set_index('branch').loc[BRANCHES]
                ax.bar(x+offset,100*g['mean_'+m],.36,yerr=100*g['std_'+m],
                       color=color,hatch=hatch,capsize=3,label=scenario)
            ax.set(xticks=x,xticklabels=BRANCHES,ylim=(0,100),ylabel=label+' (%)')
            ax.grid(axis='y',alpha=.2)
        axes[0].legend()
        fig.suptitle('Train-only model selection | mean and SD across 6 outer folds')
        show(fig,'nested_comparison','Nested baseline versus optimized procedure; three metrics with outer-fold SD.', ['nested_summary.csv'])
        '''),
        md('''
        ## Chênh lệch từng fold và cấu hình được chọn

        Mỗi điểm là (optimized − baseline) trên cùng outer test fold, đơn vị điểm phần trăm.
        Bảng bên dưới giữ tên reducer và classifier của cả 6 fold để thấy độ ổn định lựa chọn.
        '''),
        code('''
        fig, ax = plt.subplots(figsize=(10,5),layout='constrained')
        for i,branch in enumerate(BRANCHES):
            g = choices[choices.branch==branch].pivot(index='fold',columns='scenario',values='outer_f1_macro')
            delta = 100*(g.optimized-g.baseline)
            ax.scatter(i+np.linspace(-.13,.13,6),delta,label=branch,marker=['o','s','^','D'][i])
            ax.plot([i-.2,i+.2],[delta.mean()]*2,color='black',lw=2)
        ax.axhline(0,color='gray',ls='--')
        ax.set(xticks=range(4),xticklabels=BRANCHES,ylabel='F1 macro change (percentage points)',
               title='Paired outer-fold changes | black segments: mean change')
        ax.grid(axis='y',alpha=.2)
        show(fig,'nested_fold_changes','Each point is a paired fold F1 difference; no inference or confidence intervals.', ['selected_by_fold.csv'])
        chosen = choices[choices.scenario=='optimized'].copy()
        chosen['configuration'] = chosen.reducer+' / '+chosen.classifier
        display(chosen.pivot(index='branch',columns='fold',values='configuration').reindex(BRANCHES))
        '''),
        md('''
        ## ROC từ dự đoán outer OOF

        Mỗi đường gộp 258 dự đoán ngoài train. AUC hiển thị là **pooled OOF AUC**,
        có thể khác mean AUC của 6 fold. Với nested CV, classifier có thể đổi giữa fold,
        nên độ hiệu chỉnh xác suất giữa fold cũng có thể khác nhau.
        '''),
        code('''
        from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
        fig, axes = plt.subplots(2,2,figsize=(11,9),layout='constrained')
        for ax,branch in zip(axes.flat,BRANCHES):
            for scenario,color,style in [('baseline','#555555','--'),('optimized','#0072B2','-')]:
                d = oof[(oof.branch==branch)&(oof.scenario==scenario)]
                assert len(d)==258 and d.participant_id.is_unique
                assert d.probability.between(0,1).all()
                fpr,tpr,_ = roc_curve(d.label,d.probability)
                auc = roc_auc_score(d.label,d.probability)
                ax.plot(fpr,tpr,color=color,ls=style,label=f'{scenario}: AUC={auc:.3f}')
            ax.plot([0,1],[0,1],color='gray',ls=':',label='chance')
            ax.set(title=branch,xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='True positive rate')
            ax.legend(loc='lower right',fontsize=8)
        show(fig,'nested_roc','Pooled outer OOF ROC of both train-selected procedures; AUC is pooled, not mean fold AUC.', ['nested_oof.csv'])
        '''),
        md('''
        ## Confusion matrix — giữ nguyên dự đoán đã lưu

        Hiển thị số người, cùng thang màu cho cả 8 panel. Nhãn là trường `group` 0/1
        của dataset; không đổi sang cờ chẩn đoán dysgraphia. Không tối ưu ngưỡng ở đây.
        '''),
        code('''
        fig, axes = plt.subplots(4,2,figsize=(9,15),layout='constrained')
        for i,branch in enumerate(BRANCHES):
            for j,scenario in enumerate(['baseline','optimized']):
                d = oof[(oof.branch==branch)&(oof.scenario==scenario)]
                cm = confusion_matrix(d.label,d.prediction,labels=[0,1])
                disp = ConfusionMatrixDisplay(cm,display_labels=['group 0','group 1'])
                disp.plot(ax=axes[i,j],colorbar=False,cmap='Blues',values_format='d',im_kw={'vmin':0,'vmax':143})
                axes[i,j].set_title(branch+' / '+scenario)
        fig.colorbar(disp.im_,ax=axes,label='Participants',shrink=.4)
        show(fig,'nested_confusion','Counts from stored predictions, shared 0–143 color scale; no threshold tuning.', ['nested_oof.csv'])
        '''),
        md('''
        ## Độ ổn định selector và phương sai PCA

        Jaccard là trung bình trên 15 cặp tập đặc trưng của 6 outer train fold.
        Các tập train chồng lấp nên đây là chỉ báo mô tả, không phải độ tái lập độc lập.
        PCA bên dưới chỉ dùng phương sai đã fit trong outer train; không fit PCA trên toàn cohort.
        '''),
        code('''
        fig, axes = plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for i,branch in enumerate(BRANCHES):
            for method,style in [('anova','-'),('extratrees','--')]:
                d = stability[(stability.branch==branch)&stability.reducer.str.startswith(method)].sort_values('n_features')
                axes[0].plot(d.n_features,d.mean_pairwise_jaccard,ls=style,marker=['o','s','^','D'][i],label=branch+' / '+method)
            curves = []
            for fold in range(6):
                record = json.loads((OUT/'folds'/f'{branch}__fold_{fold}.json').read_text())
                curves.append(np.cumsum(record['bank']['pca_explained_variance_ratio']))
            curves = np.array(curves)
            line, = axes[1].plot(np.arange(1,curves.shape[1]+1),100*curves.mean(axis=0),label=branch)
            axes[1].fill_between(np.arange(1,curves.shape[1]+1),100*curves.min(axis=0),
                                 100*curves.max(axis=0),color=line.get_color(),alpha=.12)
        axes[0].set(xscale='log',ylim=(0,1),xlabel='Selected features (log scale)',
                    ylabel='Mean pairwise Jaccard',title='Feature selection stability')
        axes[1].set(ylim=(0,100),xlabel='PCA components',ylabel='Cumulative train variance (%)',
                    title='PCA: mean curve, shading = min–max across train folds')
        for ax in axes:
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        show(fig,'stability_and_variance','Selector Jaccard and fold-local cumulative PCA variance; shading is fold range, not CI.',
             ['feature_stability.csv']+[f'folds/{b}__fold_{f}.json' for b in BRANCHES for f in range(6)])
        display(stability.round(3))
        save_manifest('nested_manifest.json')
        '''),
        md('''
        ## Cách đọc kết quả

        - Kinematic: giảm chiều chưa tăng F1; giữ baseline nếu ưu tiên F1.
        - MOMENT: quy trình tối ưu tăng F1 nested; tiếp tục kiểm chứng, chưa khẳng định riêng ANOVA-64 tối ưu.
        - GAF/CWT: ứng viên thắng bảng thăm dò chưa giữ được lợi ích trong nested CV.
        - Những giải thích về loại nhiễu/dư thừa là giả thuyết, chưa được xác nhận bằng ablation.

        Chi tiết: `results/feature_optimization/REPORT_VI.md`. PNG/SVG và manifest nguồn
        nằm trong `results/feature_optimization/figures/`. Không thay kết quả CV hoặc notebook cũ.
        '''),
    ]
    for name,cells in [('05_feature_optimization.ipynb',overview),('06_nested_cv_and_stability.ipynb',nested)]:
        book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (.venv)','language':'python','name':'python3'}})
        nb.validate(book)
        nb.write(book,ROOT/'notebooks'/name)
        print('CREATED',name)


if __name__=='__main__':
    main()
