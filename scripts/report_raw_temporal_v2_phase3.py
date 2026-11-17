"""Actual SSL reconstruction, matched downstream and label-sensitivity results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from raw_temporal_v2_common import *
from report_raw_temporal_v2_phase0 import md


def main():
    status=json.loads((OUT/'phase3_validation.json').read_text());assert status['complete']
    config=json.loads((ROOT/'configs/raw_temporal_v2/phase3.json').read_text())
    comparison=pd.read_csv(OUT/'tables/phase3_ssl_comparison.csv');sensitivity=pd.read_csv(OUT/'tables/phase3_label_sensitivity.csv')
    all_fits=pd.read_csv(OUT/'tables/phase3_all_downstream_fits.csv');pretrain=[]
    for seed in config['seeds']:
        result=json.loads((OUT/'phase3_ssl'/f'p3_masked_ssl_seed{seed}/result.json').read_text())
        coverage=pd.read_csv(OUT/'phase3_ssl'/result['experiment_id']/'train_window_coverage.csv')
        pretrain.append(dict(seed=seed,best_epoch=result['best_epoch'],epochs=result['epochs_run'],
            seconds=result['training_seconds'],peak_gpu_mib=result['peak_gpu_mib'],
            all_train_windows=int(coverage.all_windows.sum()),eligible_train_windows=int(coverage.eligible_windows.sum()),
            **result['validation_reconstruction']))
    pretrain=pd.DataFrame(pretrain);pretrain.to_csv(OUT/'tables/phase3_reconstruction.csv',index=False)
    summary=comparison.groupby('method')[['accuracy','f1_macro','auc','brier','trainable_parameters','seconds']].agg(['mean','std'])
    flat=summary.copy();flat.columns=['_'.join(c) for c in flat.columns];flat=flat.reset_index()
    flat.to_csv(OUT/'tables/phase3_two_seed_summary.csv',index=False)
    means=comparison.groupby('method')[['f1_macro','auc']].mean()
    selected=status['selection']['selected_ssl_method'];gain=means.loc[selected,'f1_macro']-means.loc['scratch','f1_macro']
    conclusion='Có tín hiệu development thuận lợi, cần Nested CV kiểm chứng.' if gain>0 else 'SSL chưa cải thiện F1 macro so với scratch trong so sánh development này.'
    figures=[];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(name,fig,sources,description):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
        figures.append(dict(name=name,description=description,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            outputs={e:sha(OUT/'figures'/(name+'.'+e)) for e in ('png','svg')}));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));paths=[]
    for seed,ax in zip(config['seeds'],axes):
        path=OUT/'phase3_ssl'/f'p3_masked_ssl_seed{seed}/learning_curve.csv';curve=pd.read_csv(path);paths.append(path)
        for column,label,style in [('validation_masked_mse','Masked model','-'),('validation_train_constant_mse','Train-only constant','--'),('validation_interpolation_mse','Visible endpoint interpolation',':')]:
            ax.plot(curve.epoch,curve[column],label=label,linestyle=style)
        ax.set(title=f'Seed{seed}, validation72 participants',xlabel='Pretraining epoch',ylabel='Participant-balanced normalized target MSE');ax.legend(fontsize=7)
    save('p3_masked_reconstruction',fig,paths,'Validation masks fixed across epochs. Target loss excludes padding, pen-up-only target positions, invalid derivatives. MSE is in normalized signed-log input units.')
    fig,axes=plt.subplots(1,2,figsize=(11,5));names=list(dict.fromkeys(comparison.method))
    for ax,metric in zip(axes,['f1_macro','auc']):
        for seed,marker in [(42,'o'),(43,'x')]:
            a=comparison[comparison.seed==seed];ax.scatter(a[metric],[names.index(n) for n in a.method],marker=marker,label=f'Seed{seed}')
        ax.set_yticks(range(len(names)),names);ax.set_xlim(0,1);ax.invert_yaxis();ax.set_xlabel(metric+' on development validation72');ax.legend()
    save('p3_matched_downstream',fig,[OUT/'tables/phase3_ssl_comparison.csv'],'Frozen local encoder uses fixed1344-D task statistics, not a trained aggregator advertised as a linear probe. Random encoder controls share pooling and selected SSL head hyperparameters.')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,scope in zip(axes,['neural','frozen']):
        a=sensitivity[sensitivity.method.str.startswith('frozen_')] if scope=='frozen' else sensitivity[sensitivity.method.isin(['scratch','finetune'])]
        if scope=='neural':a=pd.concat([a,comparison[comparison.method.isin(['scratch','finetune'])]],ignore_index=True)
        for method,group in a.groupby('method'):
            average=group.groupby('labeled_train_participants').f1_macro.mean();ax.plot(average.index,average.values,marker='o',label=method)
            ax.scatter(group.labeled_train_participants,group.f1_macro,alpha=.6,s=15)
        ax.set(xlabel='Supervised Train participants (plus fixed72 labeled validation)',ylabel='Development macro F1',ylim=(0,1),title=scope+' label-count sensitivity');ax.legend(fontsize=7)
    save('p3_label_sensitivity',fig,[OUT/'tables/phase3_label_sensitivity.csv',OUT/'tables/phase3_ssl_comparison.csv'],'Conditional label-count sensitivity, not total annotation-budget efficiency. Fixed143 unlabeled Train pool. Frozen curves use fixed encoderseed42, linear C0.1 chosen before outcomes; neural curves have two seeds.')
    write(OUT/'figures/phase3_manifest.json',figures)
    text=f'''# Phase 3 — Masked motor self-supervised learning

Đã replay2 pretraining checkpoints,12 neural downstream fits và38 CPU readout fits. Tất cả gradient/model fits chỉ dùng participant Train của split tương ứng;43 outer Test0 không được load để phát triển. Validation72 được dùng chọn checkpoint/hyperparameters nên đây vẫn là development thăm dò.

## Objective và chống shortcut

Local encoder: `{config['pretraining']['local_encoder']}`; downstream architecture giữ đúng quyết định Phase2. Span32 sample được chọn bằng anchor pen-down, không đọc giá trị target/nhãn. Đồng thời che coordinates/pressure/orientation và velocity/speed/acceleration, cộng halo2 sample để derivative lân cận không chép lại giá trị bị che. Giữ clock, pen-state và validity flags. Decoder nhận feature map, ánh xạ đúng tâm convolution về vị trí source sample, không nén right-padding thành thời gian thật.

Target là4 kênh transformed input: source-relative x/y, pressure và speed. Loss chỉ đặt trên masked pen-down sample thật; speed cần velocity-valid. Padding, invalid derivative và pen-up-only target không tạo reward tái dựng0. Mọi eligible Train window đều được xử lý mỗi epoch, gradient trung bình theo participant; các window không có objective được đếm riêng và vẫn giữ nguyên trong classification. Online Train MSE là telemetry theo window; validation MSE trung bình theo participant. Không dùng group labels cho pretraining.

## Reconstruction và đối chứng đơn giản

{md(pretrain)}

![Masked reconstruction](figures/p3_masked_reconstruction.png)

Constant baseline chỉ fit target mean trong Train; interpolation dùng endpoint còn nhìn thấy nằm ngoài halo và hợp lệ, fallback constant nếu thiếu endpoint. MSE tốt hơn constant nhưng không hơn interpolation không đủ chứng minh encoder học đặc trưng hữu ích. Không gán đơn vị mm/lực cho các target normalized signed-log.

## Scratch, frozen linear/nonlinear và fine-tuning

{md(comparison)}

Trung bình±SD chỉ theo hai initialization seeds trên **cùng một split**, không phải sáu folds hay hai cohort độc lập:

{md(flat)}

![Matched downstream](figures/p3_matched_downstream.png)

Scratch và fine-tuning dùng cùng architecture, labeled Train IDs, optimizer/budget/early stopping. Frozen encoder là local CNN/TCN, không chứa supervised aggregator. Mean/max/std token theo14 task tạo1344 chiều cố định; linear head là Logistic Regression, nonlinear head có16 hidden units. Encoder random dùng cùng chuẩn hóa/pooling/head hyperparameters với SSL để tách tác động pretraining. Không gọi một MIL aggregator đã học nhãn là linear probe.

Best SSL method development: `{selected}`; chênh F1 macro với scratch={gain:+.6f}. {conclusion} Không bỏ kết quả âm tính. Các head settings chọn trên development phải được chọn lại bằng Train-only inner CV ở Phase4.

## Số nhãn supervised Train

{md(sensitivity[['method','seed','labeled_fraction','labeled_train_participants','accuracy','f1_macro','auc']])}

![Label sensitivity](figures/p3_label_sensitivity.png)

Đây là **sensitivity theo số nhãn Train**, không phải tổng annotation-budget efficiency: luôn có thêm72 validation labels, unlabeled Train pool143 cố định và kiến trúc đã được phát triển trước trên cohort. Subsets25%⊂50%⊂100% được stratify theo group trong Train. Frozen curves giữ encoder seed42 và linear C0.1 định trước, không chọn C bằng các full-label trial rồi gọi subset là hoàn toàn chưa biết nhãn. End-to-end curves có scratch/fine-tune khớp nhau, normalizer chỉ fit cùng143 unlabeled Train, không fit validation.

## Chi phí và chuyển Phase4

Pretraining cost được ghi riêng trong bảng reconstruction; bảng downstream ghi thời gian mỗi fit. CPU head time không bao gồm pretraining và frozen-feature extraction; thời gian extraction duy nhất nằm trong metadata từng cache, không cộng lặp cho mỗi head. Checkpoint/OOF-development predictions/config/seed/normalizer IDs được lưu riêng. Không foundation model hay full pretrained image encoder. Contrastive baseline không được lên lịch vì chưa có augmentation label-preserving đã kiểm chứng; đây là hạng mục optional không chạy, không gọi là completed.

Validation: `phase3_validation.json`; loss-mask tests kiểm tra target-copy/halo, padding/invalid derivative và gradient/coverage. Bước tiếp theo chỉ chấm candidate đã khóa bằng six-fold participant-independent nested protocol và kiểm tra complementarity. Chưa kết luận SSL tốt hơn baseline Kinematic.
'''
    (OUT/'PHASE3_REPORT_VI.md').write_text(text,encoding='utf-8',newline='\n')
    progress=json.loads((OUT/'validation.json').read_text());progress.update(phase3='complete',phase4='ready',complete=False);write(OUT/'validation.json',progress)


if __name__=='__main__':main()
