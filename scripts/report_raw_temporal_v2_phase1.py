"""Artifact-derived Phase1 report. No scientific conclusion from missing runs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from raw_temporal_v2_common import *
from report_raw_temporal_v2_phase0 import md
from train_raw_temporal_v2_phase1 import experiments


def main(figures_only=False):
    validation=json.loads((OUT/'phase1_validation.json').read_text());assert validation['complete']
    directory=OUT/'phase1_bottleneck';rows=[]
    for spec in experiments():
        r=json.loads((directory/('p1_'+spec['name'])/'result.json').read_text())
        rows.append(dict(configuration=spec['name'],family=spec['family'],**r['metrics'],
            train_f1_macro=r['train_metrics']['f1_macro'],parameters=r['parameters'],
            best_epoch=r['best_epoch'],epochs=r['epochs_run'],seconds=r['training_seconds'],
            peak_gpu_mib=r['peak_gpu_tensor_mib'],embedding_variance=r['embedding_variance']))
    table=pd.DataFrame(rows);table.to_csv(OUT/'tables/phase1_development_comparison.csv',index=False)
    table[table.configuration.str.startswith('cnn_local_')].to_csv(OUT/'tables/phase1_local_pooling_comparison.csv',index=False)
    table[table.configuration.str.startswith('task_')].to_csv(OUT/'tables/phase1_task_specific.csv',index=False)
    probes=pd.read_csv(OUT/'tables/phase1_layer_probes.csv');coverage=pd.read_csv(OUT/'tables/phase1_probe_coverage.csv')
    retention=pd.read_csv(OUT/'tables/phase1_pooling_physical_probes.csv')
    capacity=pd.read_csv(OUT/'tables/phase1_capacity_probes.csv')
    synth=pd.DataFrame([dict(test=r['kind'],pool=r['pooling'],seed=r['seed'],**r['heldout_metrics'])
        for p in sorted(directory.glob('p1_synthetic_*/result.json')) for r in [json.loads(p.read_text())]])
    synth.to_csv(OUT/'tables/phase1_pooling_synthetic.csv',index=False)
    figures=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(name,fig,sources,description):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=300,facecolor='white',bbox_inches='tight')
        figures.append(dict(name=name,description=description,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            outputs={ext:sha(OUT/'figures'/(name+'.'+ext)) for ext in ('png','svg')}));plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,8));positions=np.arange(len(table))
    ax.plot(table.f1_macro,positions,'o',label='Development validation n=72');ax.plot(table.train_f1_macro,positions,'x',label='Train n=143')
    ax.set_yticks(positions,table.configuration);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Macro F1, threshold0.5');ax.legend()
    save('p1_controlled_comparison',fig,[OUT/'tables/phase1_development_comparison.csv'],'Single development split, no CV error bars. Training/validation gap; all scheduled configurations shown.')
    pivot=probes[probes.method=='ridge'].pivot(index='level',columns='target',values='r2')
    fig,ax=plt.subplots(figsize=(12,8));im=ax.imshow(pivot.values,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=40,ha='right');ax.set_yticks(range(len(pivot)),pivot.index)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):ax.text(j,i,f'{pivot.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(pivot.iloc[i,j])>.55 else 'black')
    fig.colorbar(im,ax=ax,label='Held-out participant-balanced R2 (colors saturated outside[-1,1])')
    save('p1_layer_information',fig,[OUT/'tables/phase1_layer_probes.csv'],'Targets belong to each observation scope, not identical tasks across hierarchy. Undefined-motion complete cases excluded and counted.')
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].scatter(table.parameters,table.f1_macro);axes[0].set(xlabel='Parameter count',ylabel='Validation macro F1',ylim=(0,1))
    axes[1].scatter(table.seconds,table.peak_gpu_mib);axes[1].set(xlabel='Training + diagnostic inference seconds',ylabel='Peak allocated GPU tensor MiB')
    save('p1_resource_cost',fig,[OUT/'tables/phase1_development_comparison.csv'],'Costs include final Train/validation prediction; allocator tensor memory is distinct from nvidia-smi process memory.')
    fig,axes=plt.subplots(5,4,figsize=(16,16));curves=[];gradient_rows=[]
    for (_,r),ax in zip(table.iterrows(),axes.flat):
        path=directory/('p1_'+r.configuration)/'learning_curve.csv';curve=pd.read_csv(path);curves.append(path)
        ax.plot(curve.epoch,curve.online_train_bce,label='Online Train BCE',linestyle='--')
        ax.plot(curve.epoch,curve.validation_bce,label='Validation BCE');ax.axvline(r.best_epoch,color='#555555',linestyle=':',label='Selected epoch')
        ax.set_title(r.configuration,fontsize=8);ax.set_xlabel('Epoch');ax.set_ylabel('BCE')
        gradient_rows.append(dict(configuration=r.configuration,mean_gradient_norm=float(curve.gradient_norm_mean.mean()),embedding_variance=r.embedding_variance))
    axes[0,0].legend(fontsize=6)
    save('p1_all_learning_curves',fig,curves,'Every scheduled real-data configuration shown. Online Train BCE includes dropout and changing weights; it is not deterministic eval-Train BCE. Vertical line is BCE-selected checkpoint.')
    gradients=pd.DataFrame(gradient_rows);gradients.to_csv(OUT/'tables/phase1_gradient_representation.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,8))
    for ax,column in zip(axes,['mean_gradient_norm','embedding_variance']):
        ax.scatter(gradients[column],range(len(gradients)),marker='o');ax.set_yticks(range(len(gradients)),gradients.configuration)
        ax.invert_yaxis();ax.set_xlabel(column);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,3))
    save('p1_gradient_representation',fig,[OUT/'tables/phase1_gradient_representation.csv'],'Pre-clipping gradient norms averaged over optimizer updates/epochs; validation representation variance at selected checkpoint. Scale differences do not alone establish collapse or causality.')
    write(OUT/'figures/phase1_manifest.json',figures)
    if figures_only:return
    winner=table.sort_values(['f1_macro','auc','parameters'],ascending=[False,False,True]).iloc[0]
    # This is a frozen development decision, never an outer-Test selection.
    write(directory/'development_decision.json',dict(selected_configuration=winner.configuration,
        rule='Development macro F1, then AUC, then fewer parameters',validation_only=True,
        metrics=winner[['f1_macro','accuracy','auc']].to_dict(),scope='Exploratory; Phase2 must compare matched M1/M2/M3, not assume temporal benefit'))
    text=f'''# Phase 1 — Kiểm tra bottleneck của biểu diễn

Đã chạy và replay {validation['real_development_fits']} cấu hình trên143 Train/72 validation, cùng20 kiểm tra synthetic. Không dùng43 outer Test để chọn thiết kế. Đây là kết quả phát triển thăm dò, **không phải Nested CV hay xác nhận độc lập**.

## So sánh dự đoán và chi phí

{md(table[['configuration','accuracy','f1_macro','auc','train_f1_macro','parameters','best_epoch','seconds','peak_gpu_mib']])}

Cấu hình đứng đầu development theo quy tắc đã định trước: `{winner.configuration}`, F1 macro={winner.f1_macro:.6f}, AUC={winner.auc:.6f}. Chênh lệch trong một split không chứng minh ưu thế tổng quát. Các cấu hình chọn checkpoint bằng validation BCE, ngưỡng xác suất0.5; không chọn epoch bằng outer Test.

![Matched development comparisons](figures/p1_controlled_comparison.png)

## Probe vật lý theo tầng

Encoder v1 đóng băng được fit trên đúng143 Train; checkpoint CNN/hierarchical đã replay với dự đoán validation cũ. Ridge và RBF-Ridge chọn alpha bằng3fold participant bên trong143 Train; scaler đầu vào và target chỉ fit Train. Mỗi participant có tổng trọng số bằng nhau dù số window khác nhau. Reference là42 moment mean/std của21 input channels, chỉ dùng probe, không đưa vào classifier participant.

Các target log1p mô tả speed, pressure variability, duration, path length, width **trong chính observation đó**, cộng pen-up fraction và vị trí peak speed. Target participant là trung bình14 target task, không ghép các task thành một chuỗi thời gian giả. Những observation không có chuyển động pen-down hợp lệ không có target speed: loại complete-case, báo cáo coverage; vì vậy kết quả window/segment áp dụng cho tập con có chuyển động xác định.

{md(coverage)}

Probe các readout mới sau cùng protocol supervised, Ridge trên cùng scope window:

{md(retention[retention.method=='ridge'].pivot(index='pool',columns='target',values='r2').reset_index())}

Multi4 được nối4 token để probe128 chiều, các readout khác32 chiều. Khác biệt khả năng tuyến tính của probe chưa tách khỏi số chiều. Bảng RBF-Ridge đối chứng ở `tables/phase1_pooling_physical_probes.csv`.

Đối chứng feature-map→PCA32 có scaler/PCA fit riêng trong Train của mỗi inner split, so với pooled-window32 chiều; không PCA toàn cohort:

{md(capacity[capacity.method=='ridge'].pivot(index='level',columns='target',values='r2').reset_index())}

![Layer probes](figures/p1_layer_information.png)

Bảng đầy đủ R², MAE, correlation ở `tables/phase1_layer_probes.csv`. Không so trực tiếp R² window với task rồi suy ra lượng thông tin mất: observation và target thay đổi theo tầng. So sánh feature-map với pooled-window của cùng encoder/cùng observation phù hợp hơn, nhưng feature map544 chiều và pooled-window32 chiều nên chưa tách ảnh hưởng dung lượng probe. Padding/mask để lại dấu vết chiều dài: R² duration gần1 ở feature map không chứng minh hiểu thời gian vật lý. Pen-up fraction là thống kê trực tiếp của input flag nên R² cao cũng là kiểm tra truy xuất đơn giản. Probe cao không phải quan hệ nhân quả hay ý nghĩa lâm sàng. Validation từng được dùng early stopping encoder v1 nên vẫn là phân tích development.

## Capacity matching và kiểm tra bổ sung

Mean+max và mean+max+std được project về32 chiều, khởi tạo bằng mean; attention local khởi tạo uniform. Multi4 giữ4 nhóm vị trí convolution, dùng timestamp gốc ở tâm receptive field, context nhóm lớn gấp4 để giữ cùng số window gốc. Multi4 đồng thời thay độ hạt thời gian và context token; không quy toàn bộ thay đổi cho pooling. Số tham số, epochs và peak memory thực tế nằm trong bảng.

Mỗi aggregation ablation chỉ đổi một tầng window→stroke, stroke→task hoặc task→person; các tầng còn lại giữ MIL. Duration pooling dùng duration nguồn, không suy từ sample count. Task-specific dùng hai task xác định trước bằng tên schema, không chọn theo nhãn. CNN budget20 và learning-rate controls không phải thay đổi kiến trúc.

Task-specific ở đây mask13 task khác trong mô hình, loader vẫn xử lý bag14 task. Thời gian chạy không đại diện cho hệ thu thập chỉ một task; hai kiểm tra này không xác định số task tối thiểu cho screening.

![All learning curves](figures/p1_all_learning_curves.png)

![Gradient and representation](figures/p1_gradient_representation.png)

{md(synth.groupby(['test','pool'])[['accuracy','f1_macro','auc']].mean().reset_index())}

Synthetic rare-event/order là kiểm tra năng lực trên fixture dễ, có metadata thời gian. Nếu nhiều cấu hình cùng đạt trần thì không phân biệt được phương pháp và không chứng minh cơ chế trên dữ liệu thật.

## Kết luận có giới hạn và chuyển Phase 2

Tiny memorization Phase0 loại trừ lỗi không thể tối ưu hoàn toàn trong điều kiện chẩn đoán, nhưng không loại trừ thiếu giám sát hoặc overfit. Kết quả probe, khoảng cách Train/validation và ablation phải được đọc cùng nhau: không tự kết luận local pooling là nguyên nhân duy nhất. Khác biệt giữa các target và số chiều probe cũng là yếu tố gây nhiễu. Phase2 sẽ kiểm tra ba họ CNN đa tỉ lệ statistical/MIL/stroke Transformer dưới cùng protocol; chỉ giữ Self-Attention nếu đối chứng có bằng chứng hữu ích.

Validation: `phase1_validation.json`; ledger và checkpoint hash có trong từng thư mục. Baseline và checkpoint cũ giữ nguyên. Kết luận cuối cùng H0–H4 chờ Phase2–4.
'''
    (OUT/'PHASE1_REPORT_VI.md').write_text(text,encoding='utf-8',newline='\n')
    overall=json.loads((OUT/'validation.json').read_text());overall.update(phase1='complete',phase2='ready',complete=False)
    write(OUT/'validation.json',overall)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--figures-only',action='store_true');args=parser.parse_args()
    main(figures_only=args.figures_only)
