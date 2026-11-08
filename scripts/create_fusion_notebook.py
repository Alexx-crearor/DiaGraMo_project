"""Create a numeric-first, read-only fusion results notebook and seven figures."""
from textwrap import dedent
import nbformat as nb
from fusion_common import ROOT,OUT


def md(s): return nb.v4.new_markdown_cell(dedent(s).strip())
def code(s): return nb.v4.new_code_cell(dedent(s).strip())


def main():
    cells=[md('''
    # 08. Multimodal Fusion — bảng số cuối cùng và Nested CV

    258 participant, nhãn group, outer Stratified 6-Fold, inner 3-Fold, seed 42.
    Notebook **chỉ đọc** kết quả đã chạy, không fit hay chọn lại mô hình.
    Metric bảng số ở 0–1; delta ở điểm phần trăm; hình ở %. OOF gộp và mean±SD
    được tách cột. SD sáu fold có train chồng lấp không phải khoảng tin cậy.
    `all_selected` chọn phương pháp/tổ hợp bằng inner CV và cho phép giữ Kinematic.
    Hàng đứng đầu outer summary có thiên lệch chọn hàng thắng, không phải xác nhận độc lập.
    '''),code('''
    from pathlib import Path
    import json, hashlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    from sklearn.metrics import roc_curve, roc_auc_score

    ROOT=Path.cwd() if (Path.cwd()/'results/fusion_ablation').exists() else Path.cwd().parent
    OUT=ROOT/'results/fusion_ablation'
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation.get('numerical_validation_passed'), 'Chạy report_fusion_ablation.py trước.'
    settings=json.loads((OUT/'settings.json').read_text())
    summary=pd.read_csv(OUT/'nested_summary.csv')
    folds=pd.read_csv(OUT/'fold_scores.csv')
    oof=pd.read_csv(OUT/'nested_oof.csv')
    selected=pd.read_csv(OUT/'selected_by_fold.csv')
    ablation=pd.read_csv(OUT/'ablation_comparison.csv')
    changes=pd.read_csv(OUT/'correction_harm.csv')
    costs=pd.read_csv(OUT/'compute_costs.csv')
    METHODS=['original','compact','late','adaptive']; VARIANTS=['KM','KG','KC','KMG']
    LABELS={'original':'Original Early','compact':'Compact Early','late':'Late','adaptive':'Adaptive'}
    COLORS={'original':'#0072B2','compact':'#E69F00','late':'#009E73','adaptive':'#CC79A7'}
    core=['Kinematic_LightGBM']+[m+'__'+v for m in METHODS for v in VARIANTS]+['all_selected']
    pd.set_option('display.max_columns',None)
    pd.set_option('display.float_format','{:.6f}'.format)
    FIG=OUT/'figures'; FIG.mkdir(exist_ok=True); manifest=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def show(fig,name,description,sources):
        for ext in ('png','svg'): fig.savefig(FIG/(name+'.'+ext),dpi=180,facecolor='white')
        manifest.append(dict(name=name,description=description,sources=sources,n_participants=258,
            source_sha256={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in sources},
            uncertainty='SD of six overlapping-training outer folds, not CI',
            files={ext:hashlib.sha256((FIG/(name+'.'+ext)).read_bytes()).hexdigest() for ext in ('png','svg')}))
        plt.show(); plt.close(fig)
    print('OOF chỉ được tạo từ outer Test; lựa chọn dùng inner CV. Không thực hiện XAI.')
    '''),md('''
    ## Kết quả cuối cùng — toàn bộ bảng số Nested CV

    Kinematic_LightGBM là baseline cố định cũ. Original Early dùng embedding gốc;
    Compact/Late/Adaptive dùng GAF channel_concat (hậu tố adapter là đối chứng).
    `single__*` là đơn nhánh chọn bằng inner CV, `equal__*` chỉ cho uniform pooling.
    '''),code('''
    columns=['scenario','mean_raw_features','mean_features','accuracy','f1_macro','auc','balanced_accuracy',
             'mean_accuracy','std_accuracy','mean_f1_macro','std_f1_macro',
             'mean_auc','std_auc','mean_balanced_accuracy','std_balanced_accuracy',
             'delta_oof_accuracy_pp','delta_oof_f1_macro_pp','delta_oof_auc_pp']
    with pd.option_context('display.max_rows',None):
        display(summary[columns].sort_values('mean_f1_macro',ascending=False).reset_index(drop=True))
    '''),md('''
    ## Ablation theo nhánh và phương pháp — bảng số

    KMG so với KM đo thêm GAF; KMG so với KG đo thêm MOMENT. Mỗi quy trình được chọn
    cấu hình riêng trong Train; đây không phải hiệu ứng nhân quả của một nhánh cố định.
    Late so với equal đo giá trị của việc cho phép học trọng số. Δ ở điểm phần trăm.
    '''),code('''
    with pd.option_context('display.max_rows',None): display(ablation)
    '''),md('''
    ## Lựa chọn từng fold và các trường hợp fallback
    '''),code('''
    with pd.option_context('display.max_rows',None):
        display(selected[selected.scenario.isin(core)].reset_index(drop=True))
    '''),md('''
    ## Kết quả theo fold — bảng số đầy đủ
    '''),code('''
    with pd.option_context('display.max_rows',None):
        display(folds[folds.scenario.isin(core)].sort_values(['scenario','fold']).reset_index(drop=True))
    '''),md('''
    ## So sánh bốn phương pháp, cùng tổ hợp nhánh

    Điểm là mean F1 macro sáu outer fold, thanh là SD (không phải CI).
    Đường baseline cố định và cùng thang 0–100%; dấu vuông nhỏ là pooled OOF F1.
    '''),code('''
    base=summary.set_index('scenario').loc['Kinematic_LightGBM']
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=np.arange(4)
    for j,m in enumerate(METHODS):
        frame=summary.set_index('scenario').loc[[m+'__'+v for v in VARIANTS]]
        pos=x+(j-1.5)*.16
        axes[0].errorbar(pos,100*frame.mean_f1_macro,yerr=100*frame.std_f1_macro,fmt=['o','^','D','P'][j],capsize=3,
                         color=COLORS[m],label=LABELS[m])
        axes[0].scatter(pos,100*frame.f1_macro,marker='s',s=15,color=COLORS[m])
        axes[1].plot(x,100*frame.accuracy,marker=['o','s','^','D'][j],color=COLORS[m],label=LABELS[m])
    axes[0].axhline(base.mean_f1_macro*100,color='black',linestyle='--',label='Kinematic mean')
    axes[1].axhline(base.accuracy*100,color='black',linestyle='--',label='Kinematic OOF')
    for ax in axes: ax.set(xticks=x,xticklabels=VARIANTS,ylim=(0,100),xlabel='Modalities'); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('F1 macro (%)'); axes[0].set_title('Outer mean ± SD; squares = pooled OOF')
    axes[1].set_ylabel('Accuracy OOF (%)'); axes[1].set_title('Pooled OOF Accuracy')
    axes[0].legend(fontsize=8,loc='lower left'); axes[1].legend(fontsize=8,loc='lower left')
    show(fig,'nested_comparison','Four fusion methods and fixed Kinematic baseline; fold SD and pooled OOF.', ['nested_summary.csv'])
    '''),md('''
    ## Chênh lệch từng fold so với Kinematic

    Sáu điểm là sáu outer Test folds; các Train folds chồng lấp. Không thêm kiểm định độc lập.
    '''),code('''
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained',sharey=True)
    k=folds[folds.scenario=='Kinematic_LightGBM'].set_index('fold').sort_index()
    for ax,v in zip(axes.ravel(),VARIANTS):
        for j,m in enumerate(METHODS):
            f=folds[folds.scenario==m+'__'+v].set_index('fold').sort_index()
            ax.plot(np.arange(1,7),(f.f1_macro-k.f1_macro)*100,marker=['o','s','^','D'][j],color=COLORS[m],label=LABELS[m])
        ax.axhline(0,color='black',linestyle='--'); ax.set(title=v,xlabel='Outer fold',ylabel='Δ F1 macro (pp)',xticks=np.arange(1,7))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    show(fig,'paired_outer_folds','Paired outer fold F1 difference from fixed Kinematic.', ['fold_scores.csv'])
    '''),md('''
    ## ROC từ OOF — xem quy trình chọn phương pháp

    Chỉ hiển thị baseline và all_selected (lựa chọn bằng inner CV), không chọn đường ROC đẹp nhất.
    Xác suất pooled đến từ sáu mô hình khác nhau; AUC gộp có thể khác mean AUC từng fold.
    '''),code('''
    fig,ax=plt.subplots(figsize=(6.5,5.5),layout='constrained')
    for name,style in [('Kinematic_LightGBM','-'),('all_selected','--')]:
        frame=oof[oof.scenario==name]
        fpr,tpr,_=roc_curve(frame.label,frame.probability)
        ax.plot(fpr,tpr,linestyle=style,label=f'{name}: AUC={roc_auc_score(frame.label,frame.probability):.3f}')
    ax.plot([0,1],[0,1],':',color='gray'); ax.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1),title='Pooled OOF ROC, n=258')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    show(fig,'oof_roc','ROC for fixed baseline and nested method selection, no post-hoc curve selection.', ['nested_oof.csv'])
    '''),md('''
    ## Sửa đúng và gây hại — số người so với Kinematic

    Số dương: baseline sai, Fusion đúng. Số âm: baseline đúng, Fusion sai. Không dùng các ca Test
    này để điều chỉnh gate. Tệp prediction_changes.csv có xác suất và dự đoán từng người.
    '''),code('''
    display(changes[changes.scenario.isin(core)].reset_index(drop=True))
    names=[m+'__'+v for v in VARIANTS for m in METHODS]+['all_selected']
    frame=changes.set_index('scenario').loc[names]
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
    positions=np.arange(len(frame))
    ax.barh(positions,frame.corrected,color='#0072B2',label='Corrected',hatch='//')
    ax.barh(positions,-frame.harmed,color='#D55E00',label='Harmed',hatch='xx')
    ax.axvline(0,color='black'); ax.set(yticks=positions,yticklabels=names,xlabel='Participants relative to Kinematic',title='Corrections and harms on pooled OOF')
    ax.invert_yaxis(); ax.legend(); ax.grid(axis='x',alpha=.2)
    show(fig,'correction_harm','Counts corrected/harmed against Kinematic across all prespecified main scenarios.', ['correction_harm.csv'])
    '''),md('''
    ## Giá trị thêm từng nhánh và GAF adapter

    Heatmap thể hiện Δ OOF F1; ô nào âm được giữ nguyên, không lọc kết quả giảm.
    '''),code('''
    effects=['M','G','C','G given KM','M given KG','G concat vs adapter','G concat vs adapter given M']
    table=ablation.pivot_table(index='method',columns='effect',values='delta_oof_f1_macro_pp').reindex(index=METHODS,columns=effects)
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    limit=max(1,float(np.nanmax(np.abs(table.to_numpy()))))
    cmap=plt.get_cmap('BrBG').copy(); cmap.set_bad('#dddddd')
    im=ax.imshow(table.to_numpy(),cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
    for i in range(len(METHODS)):
        for j in range(len(effects)):
            val=table.iloc[i,j]; ax.text(j,i,'N/A' if pd.isna(val) else f'{val:+.2f}',ha='center',va='center',fontsize=9,
                                       color='white' if pd.notna(val) and abs(val)>limit*.55 else 'black')
    ax.set(xticks=range(len(effects)),xticklabels=effects,yticks=range(4),yticklabels=[LABELS[m] for m in METHODS],title='Procedure ablation: Δ pooled OOF F1 (pp)')
    ax.tick_params(axis='x',rotation=20); fig.colorbar(im,ax=ax,label='F1 macro change (pp)')
    show(fig,'modality_ablation','Matched procedure contrasts including negative effects and low-cost GAF adapter.', ['ablation_comparison.csv'])
    '''),md('''
    ## Adaptive: trọng số khi gate hoạt động và fallback từng fold

    Khi gate tắt, trọng số Kinematic bằng 1. Hình dưới dùng trọng số hiệu dụng khi gate bật;
    fallback K_only cũng bằng 1. Đây không phải giải thích XAI hoặc importance nhân quả.
    '''),code('''
    palette={'K':'#333333','M':'#E69F00','G':'#0072B2','C':'#009E73'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharey=True)
    gate_rows=[]
    for ax,v in zip(axes.ravel(),VARIANTS):
        sub=selected[selected.scenario=='adaptive__'+v].sort_values('fold')
        weights={b:np.zeros(6) for b in palette}
        for j,row in enumerate(sub.itertuples()):
            c=settings['grid'][row.candidate]
            if row.candidate=='K_only': weights['K'][j]=1
            else:
                comb=json.loads(row.combiner); weights['K'][j]=1-comb['budget']
                for b,w in zip(c['branches'][1:],comb['weights']): weights[b][j]=comb['budget']*w
            record=json.loads((OUT/'folds'/f'fold_{row.fold}.json').read_text())['results']['adaptive__'+v]
            gate_rows.append(dict(variant=v,fold=row.fold,gate_active_fraction=float(np.mean(record.get('gate_active',[False]*43))),fallback=row.candidate=='K_only'))
        bottom=np.zeros(6)
        for b,w in weights.items():
            if np.any(w): ax.bar(np.arange(1,7),w,bottom=bottom,color=palette[b],label=b); bottom+=w
        ax.set(title=v,ylim=(0,1),xticks=np.arange(1,7),xlabel='Outer fold',ylabel='Effective weight if gate active'); ax.legend(fontsize=8)
    display(pd.DataFrame(gate_rows))
    show(fig,'adaptive_weights','Gated convex contribution and Kinematic fallback in each outer fold.', ['selected_by_fold.csv','settings.json'])
    '''),md('''
    ## Số chiều và chi phí tìm kiếm đã đo

    Tổng workload có chia sẻ bank và nhiều ứng viên; không phải thời gian end-to-end hay latency.
    Trục số chiều dùng log10. Không chạy encoder/GPU trong thực nghiệm này.
    '''),code('''
    display(costs.groupby('stage',sort=False).agg(measured_fit_transform_seconds=('seconds','sum')).reset_index())
    display(pd.read_csv(OUT/'confidence_ablation.csv'))
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for m in METHODS:
        frame=summary[summary.scenario.isin([m+'__'+v for v in VARIANTS])]
        axes[0].scatter(frame.mean_features,frame.mean_f1_macro*100,label=LABELS[m],color=COLORS[m],marker={'original':'o','compact':'s','late':'^','adaptive':'D'}[m])
    axes[0].axhline(base.mean_f1_macro*100,color='black',linestyle='--'); axes[0].set(xscale='log',ylim=(0,100),xlabel='Mean classifier input dimensions (log10)',ylabel='Mean outer F1 macro (%)',title='Dimensions and mean F1')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    def family(s):
        if s.startswith('bank_'): return 'Shared branch transforms'
        if s.startswith('original__'): return 'Original Early grid'
        if s.startswith('compact__'): return 'Compact Early grid'
        return 'Branch classifier grid'
    sums=costs.assign(family=costs.component.map(family)).groupby('family').seconds.sum()/60
    axes[1].barh(sums.index,sums.values,color='#0072B2'); axes[1].set(xlabel='Measured cumulative minutes',title='Search workload; two concurrent outer jobs')
    axes[1].grid(axis='x',alpha=.2)
    show(fig,'compute_cost','Classifier dimensions versus F1 and cumulative measured search components; excludes encoders and meta fitting.', ['nested_summary.csv','compute_costs.csv'])
    '''),md('''
    ## Kiểm chứng và kết luận

    Đọc REPORT_VI.md để phân biệt cải thiện mô tả với bằng chứng xác nhận. Thiết kế tham khảo
    kết quả cũ trên cùng cohort, nên cần external holdout cho kết luận tổng quát.
    Notebook không chọn cấu hình mới và không triển khai XAI.
    '''),code('''
    (FIG/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    display(pd.DataFrame([{k:v for k,v in validation.items() if k!='numerical_artifact_sha256'}]))
    print((OUT/'REPORT_VI.md').read_text(encoding='utf-8').split('## Thiết kế')[0])
    ''')]
    notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (DiaGraMo)','language':'python','name':'python3'}})
    nb.validate(notebook)
    nb.write(notebook,ROOT/'notebooks/08_fusion_ablation.ipynb')
    print('Created notebooks/08_fusion_ablation.ipynb')


if __name__=='__main__': main()
