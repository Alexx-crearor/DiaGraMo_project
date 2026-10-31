"""Generate tables, scientific figures, a Vietnamese report, and runnable notebooks."""
import sys,json,hashlib,subprocess,platform
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,confusion_matrix,accuracy_score,f1_score,roc_auc_score
from diagramo.features import gaf,cwt,resample

def table(frame):
    cols=list(frame.columns)
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.2f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)

def main():
    generated_at=datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds')
    cv=ROOT/'results/cv'; out=ROOT/'results'; figures=out/'figures'; figures.mkdir(exist_ok=True)
    summary=pd.read_csv(cv/'summary.csv'); audit=json.loads((out/'audit/summary.json').read_text())
    expected=set(['kinematic','moment']+[p+'_'+e for p in ['gaf','cwt'] for e in ['resnet50','efficientnet_b4','vit_b_16','vgg16']])
    from itertools import combinations
    expected.update('+'.join(c) for n in [2,3,4] for c in combinations(['Kinematic','MOMENT','GAF','CWT'],n))
    classifier_names={'LogisticRegression','SVM','RandomForest','XGBoost','LightGBM','AdaBoost','MLP','Dummy'}
    complete=(len(summary)==168 and set(summary.representation)==expected
              and all(set(g.classifier)==classifier_names for _,g in summary.groupby('representation')))
    checks=[]
    assignments=pd.read_csv(cv/'fold_assignments.csv'); ids=assignments.participant_id.to_numpy()
    for p in cv.glob('*.oof.csv'):
        d=pd.read_csv(p,float_precision='round_trip')
        if len(d)!=258 or d.participant_id.duplicated().any() or not np.array_equal(d.participant_id.to_numpy(),ids): raise ValueError('Invalid OOF: '+str(p))
        if not d.probability.between(0,1).all(): raise ValueError('Invalid probability: '+str(p))
        if not np.array_equal(d.label,assignments.label) or not np.array_equal(d.fold,assignments.fold): raise ValueError('OOF labels/folds differ: '+str(p))
        if not d.prediction.isin([0,1]).all(): raise ValueError('Nonbinary predictions: '+str(p))
        actual={'accuracy':accuracy_score(d.label,d.prediction),'f1_macro':f1_score(d.label,d.prediction,average='macro'),
                'auc':roc_auc_score(d.label,d.probability)}
        r=json.loads(p.with_name(p.name.replace('.oof.csv','.json')).read_text())
        for key,value in actual.items():
            if not np.isclose(value,r['pooled'][key],atol=1e-10,rtol=0): raise ValueError('Metric does not match OOF: '+str(p))
        checks.append(p.name)
    single=summary[~summary.representation.str.contains('+',regex=False)]
    fusion=summary[summary.representation.str.contains('+',regex=False)]
    single.to_csv(out/'table_unimodal.csv',index=False); fusion.to_csv(out/'table_fusion.csv',index=False)
    signal_rows=single[single.representation.isin(['kinematic','moment']) & (single.classifier!='Dummy')].sort_values(['accuracy','auc'],ascending=False).groupby('representation',sort=False).head(3)
    image_rows=single[single.representation.str.startswith(('gaf_','cwt_')) & (single.classifier!='Dummy')].copy()
    image_rows['modality']=image_rows.representation.str.split('_').str[0]
    image_rows=image_rows.sort_values(['accuracy','auc'],ascending=False).groupby('modality',sort=False).head(3)
    fusion_rows=fusion[fusion.classifier!='Dummy'].sort_values(['accuracy','auc'],ascending=False).head(3)
    signal_rows.to_csv(out/'table_I_signal_top3.csv',index=False)
    image_rows.to_csv(out/'table_II_image_top3.csv',index=False)
    fusion_rows.to_csv(out/'table_III_fusion_top3.csv',index=False)
    summary[summary.classifier!='Dummy'].sort_values(['accuracy','auc'],ascending=False).groupby('representation',sort=False).head(1).to_csv(out/'best_per_representation.csv',index=False)
    primary=[('moment','LogisticRegression',65.1,65.1,70.1),('moment','RandomForest',65.1,63.3,71.4),('moment','SVM',70.2,69.9,74.9),
             ('kinematic','XGBoost',82.2,81.9,89.9),('kinematic','LightGBM',82.9,82.8,90.5),('kinematic','AdaBoost',83.7,83.6,90.1),
             ('gaf_efficientnet_b4','XGBoost',65.9,65.3,71.8),('gaf_vgg16','RandomForest',68.6,67.0,72.2),
             ('gaf_vit_b_16','LightGBM',73.3,73.0,78.5),('cwt_vgg16','SVM',66.3,66.0,70.8),('cwt_vit_b_16','SVM',67.1,66.4,72.1),
             ('cwt_efficientnet_b4','LightGBM',71.7,71.5,75.3),
             ('Kinematic+GAF','AdaBoost',84.9,84.9,91.1),('Kinematic+MOMENT+GAF','AdaBoost',85.3,85.2,90.8),
             ('Kinematic+MOMENT+GAF+CWT','AdaBoost',83.7,83.7,90.1)]
    comparison=[]
    for rep,clf,pa,pf,pu in primary:
        r=summary[(summary.representation==rep)&(summary.classifier==clf)]
        if r.empty: continue
        r=r.iloc[0]
        comparison.append(dict(representation=rep,classifier=clf,paper_acc=pa,local_acc=100*r.mean_accuracy,
                              delta_acc_pp=100*r.mean_accuracy-pa,paper_f1_unspecified=pf,local_f1_macro=100*r.mean_f1_macro,
                              paper_auc=pu,local_mean_auc=100*r.mean_auc,local_pooled_auc=100*r.auc))
    compare=pd.DataFrame(comparison); compare.to_csv(out/'comparison_paper.csv',index=False)
    published_complete=len(compare)==len(primary)
    three=compare[(compare.representation=='Kinematic+MOMENT+GAF') & (compare.classifier=='AdaBoost')]
    if len(three):
        t=three.iloc[0]
        replication_note=(f"Cấu hình trọng tâm Kinematic + MOMENT + GAF / AdaBoost đạt {t.local_acc:.2f}% accuracy trong bản dựng này, "
                          f"so với {t.paper_acc:.1f}% được công bố (chênh {t.delta_acc_pp:.2f} điểm phần trăm). "
                          "**Chưa tái hiện được kết quả fusion trọng tâm của tác giả.** Các khác biệt về cohort/nhãn, bộ 74 đặc trưng, "
                          "biến thể AdaBoost, checkpoint và phép biến đổi ảnh cần được đối chiếu với mã nguồn hoặc cấu hình gốc.")
    else: replication_note='Cấu hình fusion trọng tâm chưa chạy xong.'
    sensitivity_path=out/'sensitivity/summary.csv'
    if sensitivity_path.exists():
        sensitivity=pd.read_csv(sensitivity_path)[['scenario','classifier','n','accuracy','f1_macro','auc']]
        sensitivity[['accuracy','f1_macro','auc']]*=100
        sensitivity_text=table(sensitivity)
    else: sensitivity_text='Chưa chạy phân tích nhạy cảm.'
    figmeta=[]
    def save(fig,name,description,sources):
        fig.savefig(figures/(name+'.png'),dpi=220,facecolor='white')
        fig.savefig(figures/(name+'.svg'),facecolor='white')
        plt.close(fig)
        figmeta.append(dict(name=name,description=description,sources=sources,dpi=220,uncertainty='where shown: sample SD across six subject-disjoint folds, not CI',n=258,unit='participant'))
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'}):
        top=summary[summary.classifier!='Dummy'].sort_values(['accuracy','auc'],ascending=False).head(12).iloc[::-1]
        fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
        labels=[r.representation+' / '+r.classifier for r in top.itertuples()]
        ax.barh(labels,top.mean_accuracy*100,xerr=top.std_accuracy*100,color='#0072B2',capsize=3)
        ax.set(xlim=(0,100),xlabel='Accuracy (%) - mean and SD across 6 folds',title='Top CV configurations (internal ranking; n=258 participants)')
        save(fig,'cv_top12','12 configurations ranked by OOF accuracy. Bars start at zero; error bars show fold SD. Ranking does not establish external performance.',['results/cv/summary.csv'])
        # Fixed reported fusion vs the fixed kinematic AdaBoost, without local tuning.
        fixed=[('kinematic','AdaBoost','Kinematic'),('Kinematic+GAF','AdaBoost','Kinematic + GAF'),('Kinematic+MOMENT+GAF','AdaBoost','Kinematic + MOMENT + GAF')]
        fig,(ax,cm_ax)=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        latest=None
        for j,(rep,clf,label) in enumerate(fixed):
            p=cv/(rep+'__'+clf+'.oof.csv')
            if not p.exists(): continue
            d=pd.read_csv(p,float_precision='round_trip'); fpr,tpr,_=roc_curve(d.label,d.probability)
            row=summary[(summary.representation==rep)&(summary.classifier==clf)].iloc[0]
            ax.plot(fpr,tpr,linestyle=['-','--',':'][j],color=['#0072B2','#D55E00','#009E73'][j],label=f'{label} (AUC={row.auc:.3f})'); latest=(d,label)
        ax.plot([0,1],[0,1],'--',color='#777777',label='Chance')
        ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1),title='Pooled out-of-fold ROC'); ax.legend(fontsize=8,loc='lower right')
        if latest:
            d,label=latest; cm=confusion_matrix(d.label,d.prediction,labels=[0,1]); cm_ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max())
            for (i,j),v in np.ndenumerate(cm): cm_ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=15)
            cm_ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Group 0','Group 1'],yticklabels=['Group 0','Group 1'],xlabel='Predicted',ylabel='Observed',title=label+' / AdaBoost')
        save(fig,'roc_confusion','Pooled OOF ROC and confusion matrix for prespecified AdaBoost configurations; ROC combines scores from six fitted models.',['results/cv/*.oof.csv'])
        s=np.load(ROOT/'data/derived/signals512.npy',mmap_mode='r')[0,3]
        g=gaf(np.stack([resample(a,224) for a in s])); w=cwt(s)
        fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
        for j,(ch,k) in enumerate([('x',0),('y',1),('pressure',2)]):
            im=axes[0,j].imshow(g[k],cmap='RdBu_r',vmin=-1,vmax=1,origin='lower'); axes[0,j].set(title=f'GAF: {ch}',xlabel='Resampled index',ylabel='Resampled index')
            im2=axes[1,j].imshow(w[k],cmap='viridis',vmin=0,vmax=1,origin='lower'); axes[1,j].set(title=f'CWT: {ch}',xlabel='Resampled index',ylabel='Log-spaced scale index')
        fig.colorbar(im,ax=axes[0,:],label='cos(angle sum)',shrink=.75); fig.colorbar(im2,ax=axes[1,:],label='Normalized log magnitude',shrink=.75)
        save(fig,'representation_examples','First included subject, big-spiral-fast task; per-image normalization and resampling defined in protocol. CWT scales are indices, not original frequencies.',['data/derived/signals512.npy'])
    (figures/'manifest.json').write_text(json.dumps(figmeta,indent=2),encoding='utf-8')
    best=summary[summary.classifier!='Dummy'].iloc[0]
    view=summary[summary.classifier!='Dummy'].head(15)[['representation','classifier','accuracy','f1_macro','auc']].copy()
    view[['accuracy','f1_macro','auc']]*=100
    txt=f'''# Báo cáo thực nghiệm DiaGraMo

Bản cập nhật: {generated_at} (Asia/Bangkok). Bảng CSV tiếp tục được cập nhật trong lúc ma trận còn chạy.

Trạng thái: **{'Đã chạy đủ 21 cấu hình × 8 bộ phân loại (168 kết quả CV, gồm Dummy).' if complete else 'Chưa đủ toàn bộ ma trận thực nghiệm; chỉ các hàng đã có kết quả dưới đây là đã chạy.'}**

Các cấu hình được công bố trong bảng I–III: **{len(compare)}/{len(primary)} đã có kết quả thực chạy**. Ma trận đầy đủ: **{len(summary)}/168**.

## Dữ liệu và đánh giá

Đã đọc {audit['files']} tệp nguồn, {audit['recordings']} bản ghi. Cohort chính: 258 người (115 group=0, 143 group=1), mỗi người 14 tác vụ, 7 kênh. Kiểm tra {len(checks)} tệp dự đoán OOF: đúng 258 ID duy nhất/tệp, cùng thứ tự người, xác suất hữu hạn trong [0,1]. Sáu fold tách người, seed=42. Tất cả phép fit chuẩn hóa đều chỉ dùng tập train.

Đây là **tái lập độc lập với giả định được công bố**, không phải xác nhận tái hiện chính xác số liệu tác giả. Xem [quy trình và các điểm chưa xác định](../EXPERIMENT_PROTOCOL.md).

## Kết quả thực chạy

Các giá trị trong bảng sau là **OOF gộp**, đơn vị %. F1 là macro; không đồng nhất chắc chắn với F1 chưa được định nghĩa trong bài báo. Bảng xếp hạng sử dụng chính CV nên không phải hiệu năng trên tập ngoài.

{table(view)}

Cấu hình đứng đầu bảng CV: **{best.representation} / {best.classifier}**, accuracy {best.accuracy*100:.2f}%, F1 macro {best.f1_macro*100:.2f}%, AUC gộp {best.auc*100:.2f}%. Không diễn giải thứ hạng này thành bằng chứng vượt bài báo khi thiết lập tái lập chưa trùng hoàn toàn.

![Các cấu hình CV](figures/cv_top12.png)

## Đối chiếu cấu hình bài báo

Bảng đối chiếu dùng mean 6 fold, AUC gộp được giữ ở cột riêng trong CSV. Delta accuracy tính theo điểm phần trăm. Không thay công thức hay chọn siêu tham số để ép khớp 85,3%.

{table(compare[['representation','classifier','paper_acc','local_acc','delta_acc_pp','paper_auc','local_mean_auc']]) if len(compare) else 'Chưa có cấu hình đối chiếu.'}

{replication_note}

![ROC và confusion matrix](figures/roc_confusion.png)

## Phân tích nhạy cảm thăm dò

Giữ cùng danh sách người trong từng fold, rồi đổi nhãn hoặc bỏ các trường hợp có cờ omit trong cả train/test tương ứng. Đây là phân tích thăm dò trên nhánh động học, không thay nhãn hay chọn cohort sau khi xem kết quả fusion. Các giá trị là OOF gộp (%).

{sensitivity_text}

## Các điểm cần giữ khi viết báo cáo khoa học

- Số 258 khớp bài báo sau khi yêu cầu đủ tác vụ **và có nhãn**, nhưng không có danh sách ID chính thức để xác nhận cùng cohort.
- Nhãn group khác cờ chẩn đoán dysgraphia ở 5 người. Có 3 người có cờ omit trong cohort. Phân tích nhạy cảm được lưu riêng nếu đã chạy, không thay kết quả chính.
- Định nghĩa 74 đặc trưng, scale CWT, chuẩn hóa và kernel 7 kênh, phiên bản encoder, tham số classifier và loại F1 là các lựa chọn tái lập được ghi rõ trong protocol.
- Các timestamp không tăng được loại chỉ khi tính đạo hàm; không sửa tệp nguồn. Không lấy tần số 167 Hz thay cho timestamp thực tế trong công thức động học.
- Encoder được đóng băng trước CV. Không dùng nhãn/điểm nhận thức làm đầu vào.
- Chưa có kiểm định trên tập dữ liệu độc lập. Không có kết luận chẩn đoán lâm sàng từ thực nghiệm này.

## Tệp bàn giao

- `cv/summary.csv`: toàn bộ kết quả, cả OOF và mean±SD.
- `cv/*.oof.csv`: dự đoán từng người ngoài tập train.
- `cv/fold_assignments.csv`: danh sách fold cố định.
- `table_unimodal.csv`, `table_fusion.csv`, `comparison_paper.csv`.
- `audit/`: danh sách giữ/loại, kiểm tra bản ghi, checksum nguồn.
- `figures/`: biểu đồ PNG/SVG và provenance.
- `../notebooks/`: notebook xem dữ liệu, đặc trưng và kết quả; lệnh huấn luyện chạy bằng script để không phụ thuộc trạng thái kernel.

![Ví dụ các biến đổi](figures/representation_examples.png)
'''
    (out/'REPORT_VI.md').write_text(txt,encoding='utf-8')
    (out/'validation.json').write_text(json.dumps(dict(complete=bool(complete),published_configurations_complete=bool(published_complete),published_configurations_finished=len(compare),published_configurations_expected=len(primary),expected_results=168,actual_results=len(summary),oof_files_validated=len(checks),expected_representations=sorted(expected)),indent=2),encoding='utf-8')
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (ROOT/'requirements-lock.txt').write_text(freeze.stdout,encoding='utf-8')
    versions=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__)
    (out/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    if '--no-notebooks' not in sys.argv: make_notebooks()
    print('REPORT COMPLETE; full matrix:',complete,flush=True)

def make_notebooks():
    import nbformat as nb
    setup="""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src'))
print('Project:', ROOT)
"""
    notebooks={
        '02_kinematic_features.ipynb':[
            nb.v4.new_markdown_cell('# 02. Đặc trưng động học\n\n74 đặc trưng/tác vụ, 14 tác vụ/người. Công thức là bản dựng được công bố rõ, vì bài báo chưa nêu từng công thức. Chạy lại trích xuất: `.venv/Scripts/python.exe scripts/prepare.py`.'),
            nb.v4.new_code_cell(setup),
            nb.v4.new_code_cell("d=np.load(ROOT/'data/derived/kinematic.npz')\nschema=json.loads((ROOT/'data/derived/schema.json').read_text())\nprint('X:',d['X'].shape,'labels:',np.unique(d['y'],return_counts=True))\nprint('Finite:',np.isfinite(d['X']).all())\nprint('Features per task:',len(schema['features_per_task']))\nprint('Task order:',schema['tasks'])"),
            nb.v4.new_code_cell("pd.DataFrame({'feature':schema['features_per_task']})"),
            nb.v4.new_code_cell("pd.read_csv(ROOT/'results/audit/participants.csv').groupby(['included','group'],dropna=False).size().to_frame('participants')")],
        '03_pretrained_representations.ipynb':[
            nb.v4.new_markdown_cell('# 03. MOMENT, GAF và CWT\n\nEncoder tiền huấn luyện được đóng băng. Đầu vào ảnh 7×224×224. Các giả định 7 kênh và CWT được ghi trong `EXPERIMENT_PROTOCOL.md`. Chạy toàn bộ: `.venv/Scripts/python.exe scripts/run_all.py --skip-download`.'),
            nb.v4.new_code_cell(setup),
            nb.v4.new_code_cell("rows=[]\nfor p in sorted((ROOT/'data/derived').glob('*.npz')):\n    d=np.load(p)\n    rows.append({'representation':p.stem,'shape':str(d['X'].shape),'finite':bool(np.isfinite(d['X']).all()),'subjects':len(d['ids'])})\npd.DataFrame(rows)"),
            nb.v4.new_code_cell("from IPython.display import Image,display\ndisplay(Image(filename=str(ROOT/'results/figures/representation_examples.png')))"),
            nb.v4.new_code_cell("pd.DataFrame([json.loads(p.read_text()) for p in sorted((ROOT/'data/derived').glob('*.metadata.json'))])")],
        '04_evaluation_and_fusion.ipynb':[
            nb.v4.new_markdown_cell('# 04. Đánh giá 6 fold và fusion\n\nMột hàng/người; không chia ngẫu nhiên theo tác vụ. Cùng fold cho tất cả nhánh. Imputer/scaler chỉ fit trên train. F1 mặc định trong báo cáo là macro. Các bảng xếp hạng là CV nội bộ, chưa phải đánh giá ngoài độc lập.'),
            nb.v4.new_code_cell(setup),
            nb.v4.new_code_cell("summary=pd.read_csv(ROOT/'results/cv/summary.csv')\nprint('CV configurations:',len(summary))\nsummary.head(20)"),
            nb.v4.new_code_cell("folds=pd.read_csv(ROOT/'results/cv/fold_assignments.csv')\nassert folds.participant_id.is_unique\npd.crosstab(folds.fold,folds.label)"),
            nb.v4.new_code_cell("pd.read_csv(ROOT/'results/comparison_paper.csv')"),
            nb.v4.new_code_cell("from IPython.display import Image,display\ndisplay(Image(filename=str(ROOT/'results/figures/cv_top12.png')))\ndisplay(Image(filename=str(ROOT/'results/figures/roc_confusion.png')))"),
            nb.v4.new_code_cell("p=ROOT/'results/sensitivity/summary.csv'\npd.read_csv(p) if p.exists() else 'Phân tích nhạy cảm chưa chạy'"),
            nb.v4.new_code_cell("json.loads((ROOT/'results/validation.json').read_text())")]
    }
    # Complete original inspection notebook while preserving its original cells.
    p=ROOT/'notebooks/01_data_inspection.ipynb'; original=nb.read(p,as_version=4)
    marker='## Kiểm tra hoàn chỉnh từ pipeline tái lập'
    original.cells=[c for c in original.cells if c.get('metadata',{}).get('diagramo_generated') is not True]
    extra=[nb.v4.new_markdown_cell(marker+'\n\n294 tệp nguồn không đồng nghĩa 294 người đủ điều kiện. Phân biệt đủ 14 tác vụ, có nhãn và các cờ loại.'),nb.v4.new_code_cell(setup),
           nb.v4.new_code_cell("print(json.dumps(json.loads((ROOT/'results/audit/summary.json').read_text()),indent=2))\naudit=pd.read_csv(ROOT/'results/audit/participants.csv')\naudit.groupby(['task_count','group'],dropna=False).size().to_frame('participants')"),
           nb.v4.new_code_cell("r=pd.read_csv(ROOT/'results/audit/recordings.csv')\nprint('Nonfinite samples:',r.nonfinite.sum())\nprint('Timestamp steps not increasing:',r.nonincreasing_timestamps.sum())\nprint('Metadata sample count mismatches:',(r.n_samples!=r.meta_samples).sum())\naudit[audit.included & (audit.group!=audit.diagnosis_dysgraphia)][['participant_id','group','diagnosis_dysgraphia']]" )]
    for c in extra: c.metadata['diagramo_generated']=True
    original.cells.extend(extra); notebooks['01_data_inspection.ipynb']=original.cells
    for name,cells in notebooks.items():
        n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (.venv)','language':'python','name':'python3'}})
        nb.write(n,ROOT/'notebooks'/name)

if __name__=='__main__': main()
