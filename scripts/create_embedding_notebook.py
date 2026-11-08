"""Create the embedding-ablation viewer with numeric tables before figures."""
from textwrap import dedent
import nbformat as nb
from embedding_common import ROOT


def md(s): return nb.v4.new_markdown_cell(dedent(s).strip())
def code(s): return nb.v4.new_code_cell(dedent(s).strip())


def main():
    cells=[md('''
    # 07. Embedding Extraction — số liệu, Ablation và Nested CV

    **258 participant, group, Stratified 6-Fold, seed 42.** GAF/ViT-B16 và CWT/EfficientNet-B4.
    Notebook chỉ đọc artifact đã chạy, không fit mô hình hoặc chọn cấu hình mới.
    Bảng số giữ metric ở **0–1**, biểu đồ dùng %. `mean_*` là trung bình 6 fold,
    các cột metric không có tiền tố là pooled OOF. SD giữa fold không phải CI.
    Kết quả grid là thăm dò; ưu tiên nested CV để đánh giá quy trình lựa chọn.
    '''),code('''
    from pathlib import Path
    import json, hashlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

    ROOT=Path.cwd() if (Path.cwd()/'data').exists() else Path.cwd().parent
    OUT=ROOT/'results/embedding_ablation'
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation.get('numerical_validation_passed'), 'Chạy report/validation số liệu trước.'
    summary=pd.read_csv(OUT/'summary.csv')
    nested=pd.read_csv(OUT/'nested_summary.csv')
    delta=pd.read_csv(OUT/'nested_comparison.csv')
    matched=pd.read_csv(OUT/'ablation_matched.csv')
    selected=pd.read_csv(OUT/'selected_by_fold.csv')
    oof=pd.read_csv(OUT/'nested_oof.csv')
    costs=pd.read_csv(OUT/'extraction_costs.csv')
    BRANCHES=['GAF','CWT']
    MODES=['original','group_adapter','channel_mean','channel_concat']
    METRICS=[('accuracy','Accuracy'),('f1_macro','F1 macro'),('auc','ROC-AUC')]
    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_colwidth',None)
    pd.set_option('display.float_format','{:.6f}'.format)
    FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
    manifest=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def show(fig,name,description,sources,n_participants=258):
        for ext in ('png','svg'): fig.savefig(FIG/(name+'.'+ext),dpi=180,facecolor='white')
        manifest.append(dict(name=name,description=description,sources=sources,n_participants=n_participants,
            source_sha256={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in sources}))
        plt.show(); plt.close(fig)
    print('192 cấu hình; metric được kiểm tra lại từ OOF. Không thực hiện Fusion.')
    '''),md('''
    ## Kết quả cuối cùng — bảng số Nested CV

    `original_raw`: inner CV chọn classifier trên embedding gốc;
    `original_compact`: giữ nguyên quy trình 54 ứng viên giảm chiều cũ;
    các mode mới có đối chứng raw và compact riêng. `new_selected` chọn giữa 72 ứng viên mới;
    `all_selected` cho phép cả 54 ứng viên cũ thắng. Không xếp hạng outer test để chọn cấu hình.
    '''),code('''
    columns=['branch','scenario','mean_features','accuracy','f1_macro','f1_weighted','f1_binary','auc',
             'balanced_accuracy','mean_accuracy','mean_f1_macro','mean_auc','std_accuracy','std_f1_macro','std_auc']
    nested[columns].sort_values(['branch','scenario']).reset_index(drop=True)
    '''),md('''
    ## Chênh lệch với baseline và nghiên cứu giảm chiều cũ

    Mọi cột delta là **điểm phần trăm**. Đây là chênh lệch giữa các quy trình được chọn
    bên trong train, đánh giá trên cùng outer test participant.
    '''),code('''
    delta.sort_values(['branch','scenario']).reset_index(drop=True)
    '''),md('''
    ## Bảng Ablation cấu hình cố định — đủ 192 hàng

    Lưới chung gồm 8 reducer × 4 mode × 3 classifier × 2 nhánh. Xếp theo mean F1 macro.
    `original` OOF giữ nguyên từ đợt trước; các ứng viên ngoài lưới chung của đợt trước
    vẫn được giữ cho quy trình `original_compact`. Không coi bảng thắng này là xác nhận độc lập.
    '''),code('''
    fixed_columns=['branch','mode','reducer','classifier','raw_features','mean_features',
                   'accuracy','f1_macro','auc','mean_accuracy','mean_f1_macro','mean_auc',
                   'std_accuracy','std_f1_macro','std_auc','mean_classifier_fit_seconds']
    with pd.option_context('display.max_rows',None):
        display(summary.sort_values(['mean_f1_macro','mean_accuracy','mean_auc'],ascending=False)[fixed_columns].reset_index(drop=True))
    '''),md('''
    ## Tác động riêng của adaptation và giảm chiều — cùng classifier

    `delta_adaptation_*_pp`: so với original cùng reducer và classifier.
    `delta_reduction_*_pp`: so với cùng mode và classifier, không giảm chiều.
    Đây là phép so sánh matched; không gán lợi ích đổi classifier cho adapter.
    '''),code('''
    matched_columns=['branch','mode','reducer','classifier','mean_features']
    for m,_ in METRICS:
        matched_columns += ['mean_'+m,'delta_adaptation_'+m+'_pp','delta_reduction_'+m+'_pp']
    with pd.option_context('display.max_rows',None): display(matched[matched_columns])
    '''),md('''
    ## Bảy kênh ảnh và adapter — một ví dụ input

    Cùng participant đầu tiên, tác vụ đầu tiên (RCFT recognition); không chọn ví dụ theo nhãn
    hoặc kết quả phân loại. Bảy ảnh xám dùng cùng thang 0–1. `original mean` minh họa
    phép gộp sớm tương đương lớp đầu baseline; `group RGB` là adapter cố định mới.
    Các màu RGB thể hiện nhóm tín hiệu, không phải ảnh màu tự nhiên.
    '''),code('''
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from diagramo.features import gaf, cwt, resample, CHANNELS
    signal=np.load(ROOT/'data/derived/signals512.npy',mmap_mode='r')[0,0]
    gaf_images=(gaf(np.stack([resample(a,224) for a in signal]))+1)/2
    cache=ROOT/'data/derived/cwt_cache/00000.npy'
    cwt_images=np.load(cache).astype('float32') if cache.exists() else cwt(signal)
    adapter_weights=np.array(json.loads((OUT/'embeddings/GAF.json').read_text())['adapter_weights'],dtype='float32')
    fig,axes=plt.subplots(2,9,figsize=(20,5),layout='constrained')
    for i,(branch,images) in enumerate([('GAF',gaf_images),('CWT',cwt_images)]):
        for j,name in enumerate(CHANNELS):
            axes[i,j].imshow(images[j],cmap='gray',vmin=0,vmax=1)
            axes[i,j].set_title(branch+' / '+name,fontsize=9)
        axes[i,7].imshow(images.mean(0),cmap='gray',vmin=0,vmax=1)
        axes[i,7].set_title('original mean',fontsize=9)
        axes[i,8].imshow(np.einsum('rc,chw->rhw',adapter_weights,images).transpose(1,2,0))
        axes[i,8].set_title('group RGB',fontsize=9)
        for ax in axes[i]: ax.axis('off')
    show(fig,'channel_input_examples','First participant/first task: seven grayscale maps, baseline mean and fixed RGB adapter; common grayscale 0–1.',
         ['embeddings/GAF.json','settings.json','../../data/derived/signals512.npy']+
         (['../../data/derived/cwt_cache/00000.npy'] if cache.exists() else []),n_participants=1)
    '''),md('''
    ## Nested CV — Accuracy, F1 macro và AUC

    Sáu outer fold, mean ± SD; không phải CI. Mọi cách lấy biểu diễn và budget được
    chọn bằng inner CV. Cùng thang 0–100% để so sánh trực tiếp các panel.
    '''),code('''
    scenarios=['original_raw','original_compact','group_adapter_raw','group_adapter_compact',
               'channel_mean_raw','channel_mean_compact','channel_concat_raw','channel_concat_compact',
               'new_selected','all_selected']
    fig,axes=plt.subplots(2,3,figsize=(18,11),layout='constrained')
    for i,branch in enumerate(BRANCHES):
        g=nested[nested.branch==branch].set_index('scenario').loc[scenarios]
        for ax,(m,label) in zip(axes[i],METRICS):
            ax.barh(scenarios,100*g['mean_'+m],xerr=100*g['std_'+m],capsize=2,
                    color=['#555555']*2+['#0072B2']*2+['#D55E00']*2+['#009E73']*2+['#CC79A7']*2)
            ax.invert_yaxis(); ax.set(xlim=(0,100),xlabel=label+' (%)',title=branch)
            ax.grid(axis='x',alpha=.2)
    show(fig,'nested_metrics','Nested procedures: three metrics, mean and SD across six outer folds.', ['nested_summary.csv'])
    '''),md('''
    ## Chênh lệch từng outer fold

    Mỗi điểm là F1 của quy trình trừ F1 `original_raw` trên cùng 43 participant test.
    Vạch đen là mean, không phải CI. Không dùng phép kiểm định độc lập trên các fold.
    '''),code('''
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    compare_scenarios=['original_compact','group_adapter_compact','channel_mean_compact',
                       'channel_concat_compact','new_selected','all_selected']
    for ax,branch in zip(axes,BRANCHES):
        g=selected[selected.branch==branch].pivot(index='fold',columns='scenario',values='outer_f1_macro')
        for j,scenario in enumerate(compare_scenarios):
            change=100*(g[scenario]-g.original_raw)
            ax.scatter(j+np.linspace(-.12,.12,6),change,marker=['o','s','^','D','v','P'][j])
            ax.plot([j-.2,j+.2],[change.mean()]*2,color='black',lw=2)
        ax.axhline(0,color='gray',ls='--')
        ax.set(title=branch,xticks=range(len(compare_scenarios)),xticklabels=compare_scenarios,
               ylabel='F1 change vs original_raw (percentage points)')
        ax.tick_params(axis='x',labelrotation=40); ax.grid(axis='y',alpha=.2)
    show(fig,'paired_outer_folds','Paired outer-fold F1 changes; each point is one fold, black line is mean.', ['selected_by_fold.csv'])
    '''),md('''
    ## Adaptation khi giữ nguyên reducer và classifier

    Heatmap hiển thị Δ F1 macro điểm phần trăm, tất cả classifier/reducer của ba mode mới,
    so với embedding gốc tại cùng ô. Thang màu đối xứng quanh 0 dùng chung giữa các panel.
    '''),code('''
    reducer_order=['none','pca_16','pca_32','pca_var95','pca_var99','anova_64','anova_256','extratrees_256']
    clf_order=['LogisticRegression','SVM','LightGBM']
    limit=max(1,float(matched.delta_adaptation_f1_macro_pp.abs().max()))
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for i,branch in enumerate(BRANCHES):
        for j,mode in enumerate(MODES[1:]):
            ax=axes[i,j]
            g=matched[(matched.branch==branch)&(matched['mode']==mode)].pivot(
                index='reducer',columns='classifier',values='delta_adaptation_f1_macro_pp').reindex(index=reducer_order,columns=clf_order)
            assert not g.isna().any().any()
            im=ax.imshow(g.to_numpy(),cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
            for r in range(len(g)):
                for c in range(3):
                    v=g.iloc[r,c]; ax.text(c,r,f'{v:+.1f}',ha='center',va='center',
                                          color='white' if abs(v)>limit*.6 else 'black')
            ax.set(title=branch+' / '+mode,xticks=range(3),xticklabels=['LR','SVM','LGBM'],
                   yticks=range(len(g)),yticklabels=reducer_order)
    fig.colorbar(im,ax=axes,label='Matched F1 change (percentage points)',shrink=.6)
    show(fig,'adaptation_matched','Adaptation-only matched F1 deltas for every shared reducer and classifier.', ['ablation_matched.csv'])
    '''),md('''
    ## ROC OOF và confusion matrix

    ROC dùng pooled OOF AUC, không phải mean AUC các fold. Các quy trình có thể dùng
    classifier khác nhau giữa fold, nên độ hiệu chỉnh xác suất giữa fold có thể khác nhau.
    Confusion matrix dùng đúng dự đoán đã lưu, không chọn threshold từ test.
    '''),code('''
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,branch in zip(axes,BRANCHES):
        for scenario,style in zip(['original_raw','original_compact','new_selected','all_selected'],['--',':','-','-.']):
            d=oof[(oof.branch==branch)&(oof.scenario==scenario)]
            assert len(d)==258 and d.participant_id.is_unique
            fpr,tpr,_=roc_curve(d.label,d.probability)
            ax.plot(fpr,tpr,ls=style,label=f'{scenario}: {roc_auc_score(d.label,d.probability):.3f}')
        ax.plot([0,1],[0,1],color='gray',ls=':')
        ax.set(title=branch,xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='True positive rate')
        ax.legend(fontsize=8,loc='lower right')
    show(fig,'nested_roc','Pooled OOF ROC; exact original, prior compact and train-selected new/all procedures.', ['nested_oof.csv'])

    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for i,branch in enumerate(BRANCHES):
        for j,scenario in enumerate(['original_raw','new_selected','all_selected']):
            d=oof[(oof.branch==branch)&(oof.scenario==scenario)]
            disp=ConfusionMatrixDisplay(confusion_matrix(d.label,d.prediction,labels=[0,1]),display_labels=['group 0','group 1'])
            disp.plot(ax=axes[i,j],cmap='Blues',colorbar=False,values_format='d',im_kw={'vmin':0,'vmax':143})
            axes[i,j].set_title(branch+' / '+scenario)
    fig.colorbar(disp.im_,ax=axes,label='Participants',shrink=.5)
    show(fig,'nested_confusion','Confusion matrices from stored labels/predictions; shared count scale 0–143.', ['nested_oof.csv'])
    '''),md('''
    ## Chi phí và số chiều

    Cùng 32 tác vụ, batch 16, warmup và ba lần đo: median throughput, gồm transfer và aggregation,
    không gồm tạo ảnh, load model, reducer hay classifier. Đây không phải latency end-to-end.
    Peak allocated của PyTorch không phải tổng VRAM của hệ thống.
    '''),code('''
    display(costs)
    cv_jobs=pd.read_csv(OUT/'cv_job_costs.csv')
    display(cv_jobs.groupby(['branch','mode']).seconds.agg(['mean','min','max']))
    print('CV job seconds: 24 candidates x (3 inner splits + 1 outer fit); not single-model latency.')
    fit_costs=pd.read_csv(OUT/'fit_costs.csv')
    display(fit_costs.groupby(['branch','mode','split'])[['bank_fit_seconds','bank_transform_seconds']].mean())
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for branch,marker in [('GAF','o'),('CWT','s')]:
        d=costs[costs.branch==branch].set_index('mode').loc[MODES]
        for ax,column,label in zip(axes,['median_seconds','peak_gpu_allocated_mb','raw_features'],
                                   ['Seconds / 32 tasks','Peak allocated GPU (MiB)','Raw feature dimensions']):
            ax.plot(range(4),d[column],marker=marker,label=branch)
            ax.set(xticks=range(4),xticklabels=MODES,ylabel=label)
            ax.tick_params(axis='x',labelrotation=30); ax.grid(alpha=.2)
    axes[0].legend()
    show(fig,'compute_and_dimensions','Matched warm encoder benchmark: first 32 tasks from three participants, GPU allocation and raw dimensions; excludes classifier/reducer.', ['extraction_costs.csv'],n_participants=3)
    '''),md('''
    ## Cấu hình chọn ở từng fold và kiểm tra leakage

    Không suy luận “một cấu hình tối ưu đã được xác nhận” từ một quy trình chọn khác nhau giữa fold.
    Dữ liệu chi tiết train/test IDs, inner fit IDs và lựa chọn được lưu trong `folds/`, `selections/`.
    '''),code('''
    chosen=selected[selected.scenario.isin(['new_selected','all_selected'])].copy()
    chosen['configuration']=chosen['mode']+' / '+chosen.reducer+' / '+chosen.classifier
    display(chosen[['branch','scenario','fold','configuration','n_features','outer_accuracy','outer_f1_macro','outer_auc']])
    display(pd.read_csv(OUT/'selection_frequency.csv'))
    display(pd.read_csv(OUT/'fusion_candidates.csv'))
    display(pd.Series({k:v for k,v in validation.items() if k not in ('complete','stage','numerical_artifact_sha256')}))
    print('Trạng thái bàn giao cuối cùng: xem results/embedding_ablation/validation.json.')
    (FIG/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Báo cáo tiếng Việt:',OUT/'REPORT_VI.md')
    print('Không triển khai Fusion trong notebook này.')
    ''')]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (.venv)','language':'python','name':'python3'}})
    nb.validate(book); nb.write(book,ROOT/'notebooks/07_embedding_ablation.ipynb')
    print('CREATED notebooks/07_embedding_ablation.ipynb')


if __name__=='__main__': main()
