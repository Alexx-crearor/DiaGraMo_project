"""Artifact-derived Vietnamese engineering diagnosis, with original results intact."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from raw_temporal_v2_common import *


def md(frame):
    names=list(frame.columns);lines=['| '+' | '.join(names)+' |','|'+'|'.join(['---']*len(names))+'|']
    for values in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v) for v in values)+' |')
    return '\n'.join(lines)


def main():
    assert json.loads((OUT/'phase0_validation.json').read_text())['complete']
    directory=OUT/'phase0_diagnostics';source=directory/'source_audit'
    tiny=[json.loads(path.read_text()) for path in sorted(directory.glob('p0_tiny_*/result.json'))]
    synth=[json.loads(path.read_text()) for path in sorted(directory.glob('p0_synthetic_*/result.json'))]
    rows=[dict(model=r['family'],participants=r['size'],epochs=r['epochs'],eval_train_bce=r['eval_train_bce'],
        macro_f1=r['train_metrics']['f1_macro'],embedding_variance=r['embedding_variance_mean'],
        mean_pair_distance=r['mean_pair_distance'],train_seconds=r['train_seconds'],gpu_peak_mib=r['peak_gpu_bytes']/1024**2) for r in tiny]
    tinytable=pd.DataFrame(rows);tinytable.to_csv(OUT/'tables/phase0_memorization.csv',index=False)
    synthetic=pd.DataFrame([dict(test=r['test'],model=r['family'],seed=r['seed'],
        f1_macro=r['heldout_metrics']['f1_macro'],accuracy=r['heldout_metrics']['accuracy'],auc=r['heldout_metrics']['auc'],
        stats_accuracy=r['statistics_baseline']['accuracy'],epochs=r['epochs'],best_epoch=r['best_epoch'],
        parameters=r['parameters'],train_seconds=r['train_seconds']) for r in synth])
    synthetic.to_csv(OUT/'tables/phase0_synthetic_results.csv',index=False)
    channels=pd.read_csv(source/'channel_summary.csv');angles=pd.read_csv(source/'angular_discontinuities.csv')
    descriptors=pd.read_csv(source/'descriptor_summary.csv');fidelity=json.loads((source/'validation.json').read_text())
    manifest=[];figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'rawtemporalv2phase0'})
    def save(name,fig,paths,caption):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(figdir/(name+'.'+ext),dpi=300,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
        manifest.append(dict(name=name,caption=caption,sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
            outputs={ext:sha(figdir/(name+'.'+ext)) for ext in ('png','svg')}));plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,ax in zip(tiny,axes.flat):
        p=directory/row['experiment_id']/'learning_curve.csv';curve=pd.read_csv(p)
        ax.plot(curve.epoch,curve.eval_train_bce,label='Train eval BCE');ax.plot(curve.epoch,curve.online_train_bce,alpha=.6,label='Online Train BCE')
        ax.set_title(f"{row['family']}, n={row['size']}");ax.set_xlabel('Epoch');ax.set_ylabel('BCE');ax.set_yscale('log')
    axes[0,0].legend(fontsize=7)
    save('p0_memorization_curves',fig,[directory/r['experiment_id']/'learning_curve.csv' for r in tiny],
        'Balanced tiny Train bags only, dropout/decay disabled. Perfect training F1 is optimization capacity, not generalization.')
    chosen=['pressure','speed','acceleration_x','acceleration_y','azimuth','tilt'];stages=['physical','signed_log','normalized_clipped']
    fig,axes=plt.subplots(len(chosen),3,figsize=(11,13))
    for i,channel in enumerate(chosen):
        r=channels.set_index('channel').loc[channel]
        for j,stage in enumerate(stages):
            box=dict(med=r[stage+'_sample_p50'],q1=r[stage+'_sample_p25'],q3=r[stage+'_sample_p75'],whislo=r[stage+'_sample_p01'],whishi=r[stage+'_sample_p99'],fliers=[])
            axes[i,j].bxp([box],vert=False,showfliers=False);axes[i,j].set_yticks([]);axes[i,j].set_title(channel+' / '+stage,fontsize=8)
    save('p0_channel_transformations',fig,[source/'channel_summary.csv'],
        'Per-record stratified sample quantiles, p01–p99 whiskers; not full extrema. Train-domain only. Exact counts/clipping use every sample; no sensor calibration inferred.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].barh(channels.channel,channels.clipping_fraction);axes[0].set_xlabel('Fraction beyond ±10 before clipping')
    axes[1].barh(channels.channel,channels.quantization_absolute_max);axes[1].set_xlabel('Maximum absolute signed-log float16 error')
    save('p0_clipping_and_quantization',fig,[source/'channel_summary.csv'],'Exact fractions/errors over all audited Train-domain samples; quantization error is in transformed channel units.')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    gradient_sources=[]
    for family,ax in zip(('cnn','tcn','hierarchical'),axes):
        p=directory/f'p0_tiny_{family}_16_seed42/gradient_updates.csv';gradient_sources.append(p);data=pd.read_csv(p)
        for module,frame in data.groupby('module'):
            s=frame.groupby('epoch').gradient_norm.mean();ax.plot(s.index,s.values,label=module)
        ax.set_yscale('symlog',linthresh=1e-6);ax.set_title(family+', tiny16');ax.set_xlabel('Epoch');ax.set_ylabel('Gradient norm after clipping');ax.legend(fontsize=6,ncol=2)
    save('p0_module_gradients',fig,gradient_sources,'Actual module gradients in optimization diagnostics, after global clip5; unused parameter counts and update norms retained separately.')
    matrix=synthetic.groupby(['model','test']).f1_macro.mean().unstack().reindex(['cnn','tcn','hierarchical'])
    fig,ax=plt.subplots(figsize=(8,3.5));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto');plt.colorbar(im,ax=ax,label='Held-out synthetic macro-F1, two seeds')
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=20,ha='right');ax.set_yticks(range(len(matrix)),matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',color='white' if matrix.iloc[i,j]<.6 else 'black')
    save('p0_synthetic_competence',fig,[OUT/'tables/phase0_synthetic_results.csv'],
        'Known-label, representation-level synthetic bags, n64 held-out bags/run; unchanged v1 components, not the full14-task model or clinical data. Seeds share fixture design, not an independent cohort.')
    write(directory/'figure_manifest.json',manifest)
    text=f'''# Phase 0 — Chẩn đoán tính hợp lệ kỹ thuật

## Phạm vi và kết luận hiện tại

Sáu kiểm tra memorization đều đạt: CNN, TCN và Hierarchical nhớ được cả tập cân bằng8 và16 người chỉ lấy từ development Train. Điều này xác nhận khả năng tối ưu trong điều kiện chẩn đoán, **không chứng minh generalization**, không loại trừ underlearning trong budget cũ và chưa xác định bottleneck ở layer nào.

Không thay đổi preprocessing, model, checkpoint, baseline Kinematic hoặc báo cáo Raw Temporal cũ. Nhánh nghiên cứu mới là `study/raw-temporal-research-v2`. Outer fold0 được dùng làm ranh giới phát triển:143 fit,72 validation và43 outer Test bị loại khỏi quyết định phát triển. Các kết quả trên cohort này vẫn mang tính thăm dò vì đã nhiều lần nghiên cứu cùng258 người.

## 1. Nguồn dữ liệu, preprocessing và sensor fidelity

Đã kiểm tra lại hash của {fidelity['original_source_files']} nguồn SVC/metadata, {fidelity['protected_baseline_files']} artifacts baseline/Task1 và {fidelity['protected_v1_source_artifacts']} source/artifacts v1. Audit channel dùng toàn bộ {fidelity['samples']:,} samples của215 outer Train participants,3010 recordings;43 outer Test không tham gia bảng phân phối dùng định hướng thiết kế. Normalizer chỉ fit143 development Train, không fit72 validation.

SVC có thứ tự cột x,y,time,pen_status,azimuth,tilt,pressure. Mapping được truy cập bằng tên, không suy từ thứ tự kênh của extractor khác. Timestamp giữ nguyên acquisition order; duplicate/tiny/negative/gap>1s tạo ranh giới, derivative chỉ hợp lệ trong các khoảng pen-down liên tiếp. Đã tính lại velocity/acceleration bằng công thức độc lập và đối chiếu cache float32 cùng raw-index/time inventory. Không áp dụng đổi mm, lực hoặc đơn vị góc khi thiếu calibration.

{md(channels[['channel','physical_mean','physical_std','physical_minimum','physical_maximum','clipping_fraction','invalid_derivative_fraction','quantization_absolute_max']])}

Các bảng `raw_sensor_by_recording.csv` và `channel_by_recording.csv` giữ phạm vi theo task/participant, zero/nonfinite counts, effective ranges, các biến đổi signed-log và normalized/clipped. Quantiles và hình boxplot lấy tối đa128 vị trí phân tầng đều trên mỗi recording; số lượng, sums, extrema, tỷ lệ clipping và quantization error được tính trên **mọi sample**, không lấy quantile mẫu làm min/max thật.

![Các biến đổi kênh](figures/p0_channel_transformations.png)

![Clipping và quantization](figures/p0_clipping_and_quantization.png)

Azimuth quan sát có {int(angles.jumps_over_180.sum())} bước nhảy lớn hơn180 source units, trong đó {int(angles.same_pen_state_jumps_over_180.sum())} giữ cùng pen state. Chu kỳ360 ở đây chỉ là chẩn đoán ứng viên wraparound dựa trên miền giá trị quan sát, chưa là chứng nhận đơn vị. Signed-log không tự làm hai đầu miền góc gần nhau. Chưa tự sửa góc hoặc kết luận wraparound là nguyên nhân giảm accuracy; cần control biểu diễn riêng ở Phase1/2.

## 2. Source consistency với Kinematic được bảo vệ

Các descriptor Kinematic được tính lại từ SVC trên nhóm Train đã chọn trước; so với cache cũ theo đúng task/feature mapping, không ghi đè cache. Duration, width, height, path length, pressure mean/std còn có phép tính độc lập để đối chiếu extractor. Bảng74 descriptor giữ riêng sai khác tuyệt đối/tương đối; tuyệt đối không dùng handcrafted vectors làm neural input.

{md(descriptors.sort_values('maximum_absolute_difference',ascending=False).head(15))}

Kinematic cũ giữ các điểm timestamp tăng và dùng gradient đi qua pen lifts; Raw Temporal chỉ dùng motion derivative pen-down có validity mask, đồng thời giữ cả các điểm timestamp lỗi và toàn bộ coverage. Vì vậy khác biệt representation không chỉ là neural architecture. Export merged trước đây còn làm tròn pressure và thiếu11 samples trong3 recordings; lịch sử đó đã được audit và giữ nguyên.

Phát hiện metadata cũ không nhất quán: `results/raw_temporal/audit/raw_svc_comparison.json` còn mô tả `primary_input` là merged JSON, trong khi `scripts/raw_temporal_common.py:recording_map` mặc định đọc SVC và cache/source checks xác nhận SVC. Đây là lỗi mô tả provenance lịch sử, chưa là bằng chứng input training sai; không sửa file được bảo vệ.

## 3. Tiny-set memorization và gradients

{md(tinytable)}

AdamW lr0,003, dropout0, weight decay0, effective batch8, micro-batch2; tối đa1000epochs, dừng khi macro-F1≥0,999 và eval Train BCE≤0,03 trong5epochs liên tiếp. Các thay đổi này chỉ dành cho chẩn đoán, không được tự coi là production protocol tối ưu. So sánh với v1 còn khác learning rate, regularization và training duration, nên không quy toàn bộ khoảng cách cho early stopping.

![Memorization](figures/p0_memorization_curves.png)

![Gradient các module](figures/p0_module_gradients.png)

Gradient đến LocalMovement và các module downstream có finite updates. Bảng `gradient_updates.csv` lưu gradient norm sau global clip5, zero fraction, nonfinite counts, unused parameters và parameter update norm. Activation mean/variance, participant embedding variance, pairwise distances và logit spread được đo riêng; không suy representation hữu ích ngoài Train chỉ từ variance khác0. Dự đoán batch1/2/8 và train/eval khi dropout0 khớp trong tolerance1e-4 logit.

## 4. Synthetic temporal competence

{md(synthetic)}

![Synthetic competence](figures/p0_synthetic_competence.png)

Synthetic Train/validation/Test dùng các generator độc lập128/64/64 bags. Hai seeds42/43; best checkpoint chọn bằng validation BCE, không Test. Rare-event thay đổi độ dài128/256/512 và vị trí event; event-order hoán đổi hai complete patches có A/B events trong cùng multiset. Mean/std baseline được kiểm tra có accuracy0,5 cho bài thứ tự. Physical-time giữ sample patterns nhưng thay elapsed-time/dt; weak-bag có nhiều segments và event thưa, không tạo giả nhãn từng window bằng cách sao chép nhãn người.

Các model ở đây ghép các component LocalMovement, TCN/TemporalBlock và Pool hiện có để kiểm tra competence; không phải toàn bộ hierarchy14 tasks. Tensors là fixture ở mức biểu diễn21 kênh, chưa mô phỏng SVC có calibration. Không gộp synthetic n với258 người thật, không dùng synthetic accuracy làm bằng chứng clinical temporal learning. Các test yếu/âm vẫn giữ trong bảng, không mặc định Self-Attention tốt hơn.

## 5. Gate và bước tiếp theo

`phase0_validation.json` kiểm tra source/normalization isolation, sáu memorization fits,24 synthetic fits, checkpoint replay và unit tests. Gate kỹ thuật đạt khi không còn lỗi implementation nền tảng chưa giải quyết; **toàn nghiên cứu Phase0–4 chưa hoàn tất**.

Phase1 phải kiểm tra local feature maps, pooled windows, strokes/tasks/person bằng frozen probes trên người held-out trong miền phát triển, rồi so sánh pooling và aggregation dưới budget được ghi trước. Khả năng memorize chỉ loại trừ một dạng hỏng tối ưu hiển nhiên; chưa bác bỏ H0/H1 hoặc xác nhận underlearning/SSL/temporal relational modeling. Chưa chạy fusion ở Phase0.
'''
    (OUT/'PHASE0_REPORT_VI.md').write_text(text,encoding='utf-8',newline='\n')
    print('PHASE0 REPORT AND FIVE FIGURE PAIRS GENERATED',flush=True)


if __name__=='__main__':main()
