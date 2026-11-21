"""Read-only scientific synthesis of the frozen experiment; no fitting or selection."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    balanced_accuracy_score, recall_score, brier_score_loss, log_loss)
from audit_locked_temporal_data import ROOT,OUT,sha


def metrics(y,p):
    pred=p>.5
    return dict(accuracy=float(accuracy_score(y,pred)),f1_macro=float(f1_score(y,pred,average='macro',zero_division=0)),
        auc=float(roc_auc_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        sensitivity=float(recall_score(y,pred,zero_division=0)),specificity=float(recall_score(y,pred,pos_label=0,zero_division=0)),
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))


def calibration(y,p,bins=10):
    """Fixed equal-width bins, includes p=0/1; empty bins remain explicitly empty."""
    index=np.minimum((np.asarray(p)*bins).astype(int),bins-1);rows=[]
    for k in range(bins):
        at=index==k;n=int(at.sum())
        mean=float(p[at].mean()) if n else np.nan;rate=float(y[at].mean()) if n else np.nan
        rows.append(dict(bin=k,left=k/bins,right=(k+1)/bins,n=n,mean_probability=mean,observed_rate=rate,
            absolute_gap=abs(mean-rate) if n else np.nan))
    table=pd.DataFrame(rows)
    return table,float((table.n*table.absolute_gap.fillna(0)).sum()/len(y))


def markdown(frame):
    def cell(value):
        if isinstance(value,(float,np.floating)):return '' if not np.isfinite(value) else f'{value:.6f}'
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
        *('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))])


def main():
    lite=ROOT/'results/raw_temporal_v2/phase4_lite'
    raw=pd.read_csv(lite/'oof_predictions.csv',float_precision='round_trip')
    data=raw[raw.model=='stroke_transformer'].sort_values('participant_id').reset_index(drop=True)
    assert len(data)>0 and data.participant_id.is_unique
    assignments=pd.read_csv(ROOT/'results/cv/fold_assignments.csv').set_index('participant_id').loc[data.participant_id]
    np.testing.assert_array_equal(data.label,assignments.label);np.testing.assert_array_equal(data.fold,assignments.fold)
    np.testing.assert_allclose(data.fixed_average_probability,.5*(data.probability+data.kinematic_probability),rtol=0,atol=1e-15)
    complete=len(data)==258 and data.fold.nunique()==6
    y=data.label.to_numpy();probabilities=dict(Kinematic=data.kinematic_probability.to_numpy(),
        Temporal=data.probability.to_numpy(),Fixed50=data.fixed_average_probability.to_numpy())
    summary=[];folds=[];bins=[]
    for name,p in probabilities.items():
        table,ece=calibration(y,p);table.insert(0,'model',name);bins.append(table)
        summary.append(dict(model=name,n=len(y),folds=data.fold.nunique(),**metrics(y,p),ece10=ece))
        for fold in sorted(data.fold.unique()):
            at=data.fold.to_numpy()==fold
            folds.append(dict(model=name,fold=fold,n=int(at.sum()),**metrics(y[at],p[at])))
    summary=pd.DataFrame(summary);folds=pd.DataFrame(folds);bins=pd.concat(bins,ignore_index=True)
    summary.to_csv(OUT/'matched_oof_metrics.csv',index=False);folds.to_csv(OUT/'matched_fold_metrics.csv',index=False)
    bins.to_csv(OUT/'calibration_bins.csv',index=False)
    foldstats=folds.groupby('model')[list(metrics(y,probabilities['Kinematic']))].agg(['mean','std'])
    foldstats.columns=['_'.join(c) for c in foldstats.columns];foldstats.reset_index().to_csv(OUT/'fold_mean_sd.csv',index=False)
    fold_delta=folds[folds.model=='Fixed50'].set_index('fold')[list(metrics(y,probabilities['Kinematic']))]-folds[folds.model=='Kinematic'].set_index('fold')[list(metrics(y,probabilities['Kinematic']))]
    fold_delta.reset_index().to_csv(OUT/'fusion_minus_kinematic_by_fold.csv',index=False)
    k=probabilities['Kinematic']>.5;t=probabilities['Temporal']>.5;f=probabilities['Fixed50']>.5
    data['kinematic_correct']=k==y;data['temporal_correct']=t==y;data['fusion_correct']=f==y
    data['fusion_effect']=np.where((k!=y)&(f==y),'corrected',np.where((k==y)&(f!=y),'harmed',np.where(k==y,'unchanged_correct','unchanged_wrong')))
    data['probability_change']=probabilities['Fixed50']-probabilities['Kinematic']
    data.to_csv(OUT/'participant_complementarity.csv',index=False)
    effect=data.groupby(['fold','fusion_effect']).size().unstack(fill_value=0).reindex(columns=['corrected','harmed','unchanged_correct','unchanged_wrong'],fill_value=0)
    effect['net_corrected']=effect.corrected-effect.harmed;effect.reset_index().to_csv(OUT/'fusion_effect_by_fold.csv',index=False)
    overlap=pd.DataFrame([dict(both_correct=int(((k==y)&(t==y)).sum()),kinematic_only=int(((k==y)&(t!=y)).sum()),
        temporal_only=int(((k!=y)&(t==y)).sum()),both_wrong=int(((k!=y)&(t!=y)).sum()),
        probability_correlation=float(np.corrcoef(probabilities['Kinematic'],probabilities['Temporal'])[0,1]))])
    overlap.to_csv(OUT/'branch_error_overlap.csv',index=False)
    # Paired, label-stratified participant bootstrap; fitted models held fixed.
    rng=np.random.default_rng(42);indices=[np.flatnonzero(y==c) for c in (0,1)];samples={name:[] for name in ('Temporal','Fixed50')}
    for _ in range(2000):
        at=np.concatenate([rng.choice(v,len(v),replace=True) for v in indices]);reference=metrics(y[at],probabilities['Kinematic'][at])
        for name in samples:
            other=metrics(y[at],probabilities[name][at]);samples[name].append({key:other[key]-reference[key] for key in reference})
    intervals=[]
    reference=metrics(y,probabilities['Kinematic'])
    for name,values in samples.items():
        observed=metrics(y,probabilities[name]);values=pd.DataFrame(values)
        for key in reference:
            low,high=np.quantile(values[key],[.025,.975]);intervals.append(dict(model=name,reference='Kinematic',metric=key,
                delta=observed[key]-reference[key],lower95=low,upper95=high,replicates=2000,
                unit='participant',stratified_by_label=True,fitted_models_fixed=True))
    intervals=pd.DataFrame(intervals);intervals.to_csv(OUT/'paired_bootstrap.csv',index=False)
    # Historical development ablations are not OOF and do not become new fits.
    ablations=[]
    for path in sorted((ROOT/'results/raw_temporal_v2/phase2_temporal').glob('p2_*/result.json')):
        result=json.loads(path.read_text());ab= result.get('architecture',{})
        ablations.append(dict(experiment=result['experiment_id'],seed=result.get('seed'),
            design=ab.get('design',ab.get('family')),relative_time=ab.get('relative_time'),relative_spatial=ab.get('relative_spatial'),
            local_encoder=ab.get('local_encoder'),parameters=result['parameters'],training_seconds=result['training_seconds'],
            best_epoch=result['best_epoch'],**{key:value for key,value in result.get('metrics',{}).items()},
            source=path.relative_to(ROOT).as_posix(),source_sha256=sha(path),scope='Historical development, not six-fold OOF'))
    ablations=pd.DataFrame(ablations);ablations.to_csv(OUT/'historical_ablation_evidence.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    colors=['#0072B2','#D55E00','#009E73'];markers=['o','s','^']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for (name,p),color,marker in zip(probabilities.items(),colors,markers):
        table=bins[bins.model==name]
        axes[0].plot(table.mean_probability,table.observed_rate,marker=marker,color=color,label=name)
        axes[1].stairs(np.histogram(p,bins=np.linspace(0,1,11))[0],np.linspace(0,1,11),color=color,label=name)
    axes[0].plot([0,1],[0,1],'k--',linewidth=1);axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted P(group=1)',ylabel='Observed fraction group=1',title=f'OOF reliability (n={len(y)}, 10 fixed bins)')
    axes[1].set(xlim=(0,1),xlabel='Predicted P(group=1)',ylabel='Participants',title='OOF probability distribution')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    for suffix in ('png','svg'):fig.savefig(OUT/f'calibration.{suffix}',dpi=300)
    plt.close(fig)
    associations=pd.read_csv(OUT/'source_associations.csv');mapping=pd.read_csv(OUT/'prefix_city_mapping.csv',keep_default_na=False)
    audit=json.loads((OUT/'data_audit.json').read_text())
    cohort=pd.read_csv(ROOT/'data/derived/participants.csv').set_index('participant_id')
    assert len(cohort)==258 and set(cohort.index)==set(raw.participant_id)
    flags=pd.crosstab(cohort.flag_to_omit_total,cohort.group).rename_axis('flag_to_omit_total').reset_index()
    flags.columns=['flag_to_omit_total','group0','group1'];flags.to_csv(OUT/'cohort_selection_audit.csv',index=False)
    diagnosis=pd.crosstab(cohort.group,cohort.diagnosis_dysgraphia).rename_axis('group').reset_index()
    diagnosis.columns=['group','diagnosis0','diagnosis1'];diagnosis.to_csv(OUT/'label_definition_crosstab.csv',index=False)
    source=pd.read_csv(OUT/'source_audit_participants.csv',keep_default_na=False).set_index('participant_id').loc[data.participant_id]
    source_rows=[]
    for city in sorted(source.id_city.unique()):
        at=source.id_city.to_numpy()==city;labels=y[at]
        for name,p in probabilities.items():
            row=dict(city=city,model=name,n=int(at.sum()),group0=int((labels==0).sum()),group1=int((labels==1).sum()),
                accuracy=float(accuracy_score(labels,p[at]>.5)),brier=float(brier_score_loss(labels,p[at])),
                auc=float(roc_auc_score(labels,p[at])) if len(np.unique(labels))==2 else np.nan)
            source_rows.append(row)
    pd.DataFrame(source_rows).to_csv(OUT/'descriptive_metrics_by_city.csv',index=False)
    main_f1=intervals[(intervals.model=='Fixed50')&(intervals.metric=='f1_macro')].iloc[0]
    bymodel=summary.set_index('model');kr=bymodel.loc['Kinematic'];tr=bymodel.loc['Temporal'];fr=bymodel.loc['Fixed50']
    runtime=pd.read_csv(lite/'runtime_memory.csv');temporal_cost=runtime[runtime.candidate=='stroke_transformer'] if 'candidate' in runtime else runtime[runtime.model=='stroke_transformer']
    sessions=[json.loads(path.read_text()) for path in sorted((lite/'sessions').glob('*.json'))]
    last_session=sessions[-1]
    status=dict(frozen_stroke_oof_complete=complete,participants=len(data),folds=int(data.fold.nunique()),
        exploratory_internal_cv=True,external_validation=False,research_complete=False,
        new_training_or_calibration_in_report=False,new_experiment_launched=False,
        original_lite_primary_remains_duration=True,locked_research_direction='Multi-scale Relative Stroke Transformer + Kinematic LightGBM',
        inputs_sha256={str(path.relative_to(ROOT)):sha(path) for path in [lite/'oof_predictions.csv',OUT/'data_audit.json',ROOT/'configs/raw_temporal_v2/phase4_lite.json',ROOT/'data/derived/participants.csv']})
    (OUT/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    report=f'''# DiaGraMo — Báo cáo nghiên cứu theo đặc tả khóa

Trạng thái: **{'Hoàn tất sáu folds của đánh giá Stroke Transformer cố định' if complete else 'Chưa đủ sáu folds; kết quả một phần'}**, {len(y)} người, {data.fold.nunique()}/6 folds. Đây là **exploratory internal CV**, không phải external validation hay bằng chứng lâm sàng. Nghiên cứu tổng thể vẫn chưa hoàn tất. Không triển khai thí nghiệm tiếp theo trong báo cáo này.

**Kết quả chính:** Kinematic F1={kr.f1_macro:.6f}, Temporal F1={tr.f1_macro:.6f}, Fixed50 F1={fr.f1_macro:.6f}. Fixed50−Kinematic={main_f1.delta:+.6f}, CI95% [{main_f1.lower95:+.6f},{main_f1.upper95:+.6f}]. {'Point estimate fusion thấp hơn baseline, không có bằng chứng cải thiện tổng thể.' if fr.f1_macro<=kr.f1_macro else 'Point estimate fusion cao hơn baseline; phải xét CI, tính nhất quán và confounding trước khi diễn giải.'} {'Tín hiệu tích cực ở hai folds đầu không được duy trì trong kết quả toàn bộ sáu folds.' if complete and fr.f1_macro<=kr.f1_macro else ''}

## 1. Kiến trúc thực sự được triển khai

Nhánh Kinematic gồm 14 tác vụ × 74 đặc trưng = 1.036, LightGBM được bảo vệ, xuất pK. Nhánh Temporal xuất pT, sử dụng tín hiệu online gốc SVC và 21 kênh tiền xử lý. Window không chồng lấp, tối đa 128 mẫu và không vượt ranh giới pen-state/timestamp. Segment là đơn vị tính toán từ các ranh giới đó, **không phải nét viết được chuyên gia gán nhãn lâm sàng**.

```mermaid
flowchart LR
    S[SVC gốc] --> P[21 kênh; windows tối đa 128 mẫu]
    P --> C[CNN kernel 5/13/29; 4 token/window]
    C --> G[Learned pooling thành segment]
    G --> A[Relative attention 2 heads; tối đa 32 segment/nhóm]
    A --> T[Task identity và learned pooling]
    T --> PT[pT]
    K[1036 Kinematic features] --> L[Protected LightGBM]
    L --> PK[pK]
    PT --> F[Trung bình cố định 50/50]
    PK --> F
```

`MultiScaleLocal` có ba Conv1d 21→8, kernel 5/13/29, stride 4; nối 24 kênh, Conv1d 1×1→32; hai residual convolution kernel 3 với dilation 2/4. GELU; giữ bốn token 32 chiều/window bằng masked mean trên bốn vùng feature map. Learned pooling gom tối đa 64 token (16 window) mỗi chunk segment; weighted scatter-average theo số token hợp lệ gom chunk về segment. Relative attention **hai heads** chỉ chạy trong nhóm tối đa **32 segment** của cùng tác vụ, không trên toàn bộ chuỗi mẫu gốc.

Relative bias: signed-log chênh lệch thời gian, positive gap, duration ratio, displacement centroid x/y, khác biệt tỷ lệ pen-down và self-identity; MLP 7→8→2. Spatial dùng device units, centroid xấp xỉ từ cache; không phải khoảng cách sinh cơ học đã hiệu chuẩn. Attention có residual, LayerNorm và FFN 32→64→32. Learned pooling gom segment chunks thành task, thêm task identity embedding, LayerNorm/Linear/GELU/dropout, rồi **learned pooling trên 14 task**, LayerNorm và linear head. **Không có task-level Transformer** và không dùng duration-weighted participant pooling của `RecoveryModel`. Tổng tham số Temporal: 25.252.

Fusion khóa: **pF = 0,5 pK + 0,5 pT**, không tham số học. Threshold giữ nguyên code `p > 0.5`. Seed42, 28 epochs cuối cùng, AdamW lr0,001/weight decay0,001, dropout0,1, dim32, effective batch8/microbatch2, gradient clip1. Không early stopping, calibration, threshold tuning hoặc chọn lại checkpoint theo outer Test. Config lite lịch sử vẫn đánh dấu Duration là primary tại thời điểm khóa ban đầu; hướng nghiên cứu mới ưu tiên Stroke theo yêu cầu người dùng sau khi đã xem partial scores. Sự thay đổi trọng tâm này không được gọi là lựa chọn độc lập với cohort.

Nguồn triển khai: [multiscale.py](../../../src/diagramo/temporal_v2/multiscale.py), [dataset.py](../../../src/diagramo/temporal/dataset.py), [frozen trainer](../../../scripts/run_raw_temporal_v2_phase4_lite.py) và [config](../../../configs/raw_temporal_v2/phase4_lite.json).

## 2. Dataset và tiền xử lý

258 participants, nhãn `group` (115 lớp0/143 lớp1), 14 tasks đầy đủ; `group` là phân nhóm của dataset, không đồng nhất với chẩn đoán dysgraphia riêng. Giữ đúng assignments Stratified6/shuffle/seed42; mỗi outer Train215/Test43, participant không giao nhau. Normalizer fit riêng 215 Train và kiểm tra từ checkpoint; ID chỉ tra cứu recording và chia fold, không phải biến đầu vào. Task ID là danh tính tác vụ, khác participant ID.

Selection lịch sử trong `prepare.py` yêu cầu đủ14 tasks, không trùng task, tín hiệu hợp lệ/hữu hạn và group∈{{0,1}}; **không dùng `flag_to_omit_total` làm điều kiện loại**. Cohort cố định hiện có3 người flag=1, vẫn được giữ để không đổi protocol/baseline. Đây là hạn chế cần xác minh với chủ dữ liệu, không tự loại người sau khi xem OOF. Chẩn đoán riêng có140 người diagnosis1/118 diagnosis0, khác số lượng group.

{markdown(flags)}

{markdown(diagnosis)}

21 kênh: relative_x/y, pressure, azimuth, tilt, elapsed_time, delta_time, pen_down, vx/vy/speed, ax/ay, velocity_valid, acceleration_valid, dt_valid, duplicate_time, negative_time, tiny_interval, large_gap, boundary. Ngưỡng dt hợp lệ [1e-4,1] theo code/đơn vị nguồn; đạo hàm chỉ qua các mẫu pen-down cùng trạng thái và dt hợp lệ. Không sort hoặc xóa mẫu để che timestamp lỗi. Signed-log cho kênh liên tục, Train normalization và clip [-10,10]. Dữ liệu RAM float16 là tối ưu lưu trữ, math/gradients float32; inventory thời gian/index được giữ. Cache hiện tại được tạo lại từ SVC gốc sau audit lịch sử; không nhầm audit merged export cũ với cache đang sử dụng. Xem Phase0 source audit và source migration có SHA để truy nguyên khác biệt export.

## 3. OOF và đối chiếu participant-matched

{markdown(summary)}

Mean ± sample SD từng fold có trong `fold_mean_sd.csv`; AUC trên pooled OOF không bằng trung bình AUC fold. Các metrics ở đây chỉ dùng đúng các participant đã có OOF Temporal. Không so partial Temporal với baseline toàn cohort.

{markdown(folds)}

## 4. Complementarity và mức bất định

{markdown(overlap)}

Fusion corrected/harmed được tính bằng so sánh nhãn dự đoán pF với pK, không phải chỉ việc xác suất tăng/giảm. Chi tiết từng participant gồm pK/pT/pF và thay đổi xác suất nằm trong `participant_complementarity.csv`.

{markdown(effect.reset_index())}

Chênh lệch Fixed50−Kinematic theo fold (Brier/log-loss âm là tốt hơn):

{markdown(fold_delta.reset_index())}

F1 tăng ở {int((fold_delta.f1_macro>0).sum())}/{len(fold_delta)} folds, giảm ở {int((fold_delta.f1_macro<0).sum())}, bằng ở {int((fold_delta.f1_macro==0).sum())}. Tính nhất quán phải xét cùng cỡ mẫu43/fold, không chỉ point estimate pooled.

{markdown(intervals[intervals.model=='Fixed50'][['metric','delta','lower95','upper95']])}

Bootstrap paired 2.000 lần, seed42, lấy lại mẫu participant trong từng lớp (cố định tỷ lệ lớp), cùng chỉ số cho cả ba mô hình; percentile95%. Giữ nguyên mô hình đã fit, không tái huấn luyện bootstrap. CI không bao gồm toàn bộ bất định do training/model selection, phụ thuộc do Train folds chồng lấp hoặc confounding nguồn; không sửa được historical researcher selection bias. Không chọn tham số hay thay rule dựa trên CI này.

## 5. Calibration

Brier/log-loss/ECE10 và reliability plot là **diagnostics** từ OOF, không dùng để hiệu chỉnh xác suất. ECE phụ thuộc binning, chỉ 258 người nên các bin ít mẫu có thể không ổn định; empty bins được giữ trong CSV và không nối như dữ liệu quan sát. Fusion có thể giúp do làm dịu overconfidence hoặc calibration, không tự động chứng minh Temporal học thêm cấu trúc vận động. Chưa có train-only calibrated Kinematic control trong thí nghiệm cố định, vì thêm nó sau khi xem Test sẽ đổi protocol. Nếu làm nghiên cứu tiếp, calibration phải fit/cross-fit hoàn toàn trong outer Train và được khóa riêng.

Fixed50 thay đổi sensitivity {fr.sensitivity-kr.sensitivity:+.6f}, specificity {fr.specificity-kr.specificity:+.6f}, Brier {fr.brier-kr.brier:+.6f}, log-loss {fr.log_loss-kr.log_loss:+.6f}, ECE10 {fr.ece10-kr.ece10:+.6f}. Đánh giá các thước đo cùng nhau; ECE giảm riêng lẻ không đủ khẳng định calibration tốt hơn nếu proper scoring rules xấu đi.

![Calibration và phân bố xác suất](calibration.png)

## 6. Data validity: prefix, nguồn thu thập và nhãn

Metadata được nối chính xác bằng ID với merged participant metadata và kiểm tra nhãn cả258 người. CSV cognitive gốc có {audit['raw_csv_exact_id_matches']} exact ID matches; group/city/site được đối chiếu giữa CSV gốc và merged metadata. Các biến nguồn chỉ dùng kiểm toán, **không huấn luyện predictor bằng ID/prefix/city/site/date**. Purity bên dưới là thống kê mô tả trên cohort, không phải accuracy cross-validation của một mô hình.

{markdown(associations)}

{markdown(mapping)}

Trong cohort đầy đủ, BR/city1 có 115 lớp0 và7 lớp1; 14 thành phố còn lại có136 người, đều lớp1. Prefix→city là ánh xạ một-một trên cohort. Cramér V≈0,947 cho city–group, và label purity mô tả≈97,29%: **source-label confounding rất mạnh**. Không suy ra city gây khác biệt chữ viết, cũng không suy ra lỗi gán nhãn. Không thể dùng thành tích CV ngẫu nhiên hiện tại để tách bệnh lý khỏi nguồn. `descriptive_metrics_by_city.csv` chỉ là kiểm tra hậu nghiệm theo nguồn, không phải một validation/site-held-out mới; city1 chỉ có7 người lớp1 nên bất định lớn, các city chỉ một lớp không có AUC hợp lệ.

Các bảng `source_label_contingencies.csv` và `source_fold_overlap.csv` cho thấy phân bố group theo prefix, city, school/site và kỳ thu thập, cũng như nguồn xuất hiện đồng thời trong Train/Test. `id_school_ppp` là trường/điểm theo tên metadata, không tự đồng nhất với hardware hoặc bệnh viện tuyển mẫu. `date_of_administration` là kỳ thu thập, không có bằng chứng đây là batch phần cứng. Các trường phần cứng/batch/tuyển mẫu được tìm thấy: {audit['hardware_batch_recruitment_fields_found']}. Không có metadata phù hợp thì nguyên nhân chính xác vẫn **chưa xác định**, không được quy kết hardware/site bằng suy đoán. Missing counts: {audit['field_missing_counts']}.

**Ba vấn đề khác nhau:** (i) confounding nguồn: tín hiệu có thể chứa khác biệt site/thiết bị/tuyển mẫu tương quan group dù ID không vào network; (ii) direct leakage: ID, group hoặc Test labels bị đưa vào input/fitting; kiểm tra pipeline hiện tại không phát hiện đường vào này, nhưng không chứng minh hết mọi confounding; (iii) researcher-level selection bias: cohort đã dùng trong nhiều quyết định nghiên cứu. Participant-independent fitting không làm các quyết định model selection trở thành validation độc lập. Nếu city/site gần như chỉ có một lớp, không thể tách ảnh hưởng nguồn và group bằng CV ngẫu nhiên; cần thiết kế thu thập hai lớp trong cùng nguồn và nguồn độc lập, không tự sửa assignments hiện tại.

## 7. Diễn giải ablation có đối chứng

`historical_ablation_evidence.csv` giữ lại 12 fits Phase2, SHA nguồn, số tham số, runtime và epoch chọn trên development. Đây là split143 Train/72 validation và chủ yếu một seed cho ablation, **không phải sáu-fold OOF**; không được gộp như sáu-fold bằng chứng. MIL và Stroke dùng cùng multi-scale local encoder/learned pooling, nhưng attention làm tăng capacity từ16.626 lên25.252 tham số; statistical có19.730, original-local MIL4.018. Vì vậy chênh lệch không tự cô lập tác dụng cấu trúc thời gian khỏi capacity/optimization.

Ở seed42 development, full relative F1≈0,7469; tắt relative time≈0,5824; tắt spatial≈0,7734; tắt cả time+spatial≈0,7884. Các kết quả không chứng minh mọi thành phần relative đều có lợi; time/space ở đây chỉ là relative bias, không xóa toàn bộ kênh thời gian/vị trí khỏi encoder. Training durations và số epoch khác nhau do development checkpoint selection; chưa có equal-budget/equal-capacity multi-seed ablation hoàn chỉnh.

Historical perturbation: transport cùng content và metadata gần như bất biến vì permutation equivariance của attention và pooling; đó không phải bằng chứng mô hình bỏ qua thời gian. Content-only reversal giữ metadata tại chỗ tạo mismatch và có thể out-of-distribution; thay đổi xác suất nhỏ không cho suy luận nhân quả hay chứng minh motor invariance. Rare-event/high-speed removal cũng thay phân bố input. Attention weights là trọng số tính toán, không phải causal explanation. Không chạy perturbation hoặc ablation mới sau khi xem frozen Test.

## 8. Quyết định C1/C2/C3

ΔF1 Fixed50−Kinematic hiện tại = {main_f1.delta:+.6f}, CI95% [{main_f1.lower95:+.6f},{main_f1.upper95:+.6f}]. {'Đủ sáu folds của protocol cố định để báo cáo toàn bộ; không tự khởi chạy nghiên cứu mới.' if complete else 'Chưa đủ sáu folds; chưa được chốt lợi ích tổng thể.'}

C1 vẫn là giả thuyết về structured temporal representation: đã triển khai cấu trúc và có development controls, nhưng chưa chứng minh đóng góp độc lập với capacity, pooling, optimization hoặc calibration. C2 chỉ có thể được mô tả ở mức kết quả complementary OOF nội bộ của frozen average; cần xem CI, corrected/harmed và tính nhất quán sáu folds, đồng thời đặt confounding nguồn lên trước F1. Ngay cả khi point estimate tăng, không gọi đây là incremental motor evidence đã được xác nhận trên dữ liệu độc lập.

**Quyết định hiện tại:** chưa đầu tư full nested architecture/fusion search hoặc SSL trên cùng cohort để chạy theo F1. Ưu tiên xác minh nguồn tuyển mẫu, điều kiện thu thập/phần cứng và bổ sung dữ liệu có cả hai lớp trong cùng nguồn. C1 có thể theo đuổi như một nghiên cứu đối chứng riêng (equal-capacity/equal-budget, multi-seed, relative-time/spatial và order stress có thảo luận OOD), không coi development ablation hiện tại là kết luận đóng góp. C2 cần source-aware/independent validation và calibrated Kinematic control trước khi tuyên bố giá trị vận động bổ sung. Cả hai hướng chỉ được **đề xuất**, chưa khởi chạy.

Nếu tín hiệu cải thiện nhất quán và không bị nguồn dữ liệu giải thích, đề xuất nghiên cứu riêng có train-only calibrated Kinematic control, cross-fitted fusion và tập kiểm chứng độc lập; chưa chạy trong nhiệm vụ này. Nếu hiệu quả yếu/không nhất quán hoặc source-label confounding mạnh, ưu tiên xác minh recruitment/site/device và validation phù hợp thay vì tăng độ phức tạp. Không bắt đầu SSL, larger Transformer, architecture grid, selective fusion hoặc threshold tuning. C3 (cross-domain/language robustness và clinical biomarkers) là hướng tương lai **chưa được chứng minh**.

## 9. Bảo toàn và chi phí

Snapshot trước khi resume ở `history_9363d15/`; checkpoints thành công không huấn luyện lại. Cấu hình/chữ ký training vẫn giữ nguyên; các aggregate outputs cập nhật được giữ cả bản snapshot. Runtime từng fit, phần dở và GPU allocated/reserved ở `../phase4_lite/runtime_memory.csv` và `incomplete_fit_runtime.csv`; session logs ghi giới hạn110 phút/một worker. Không cộng trùng thời gian fit và session, không gọi GPU tensor allocation là tổng process VRAM. Ảnh PNG300dpi/SVG và các bảng đi kèm phục vụ manuscript, chưa gắn chuẩn journal cụ thể. Không push/merge tự động.

Phiên mới nhất: {last_session.get('elapsed_seconds',0)/60:.2f} phút so với giới hạn {last_session['maximum_seconds']/60:.0f} phút; includes_process_setup={last_session.get('includes_process_setup')}, includes_reporting={last_session.get('includes_reporting',False)}. Thời gian startup trước SessionBudget không được tính như GPU training. Tổng chi phí training các Temporal folds đã hoàn tất (cộng dồn cả các phiên trước) là {temporal_cost.training_seconds.sum()/60:.2f} phút; đây không phải thời gian của riêng phiên mới. Peak tensor allocated tối đa {temporal_cost.peak_gpu_allocated_mib.max():.2f} MiB; reserved tối đa {temporal_cost.peak_gpu_reserved_mib.max():.2f} MiB. `gpu_snapshot_20261121.csv` chỉ là snapshot NVML của phiên mới, không phải peak process memory liên tục.
'''
    (OUT/'LOCKED_RESEARCH_REPORT_VI.md').write_text(report,encoding='utf-8')
    print('LOCKED SCIENTIFIC REPORT',status,flush=True)


if __name__=='__main__':main()
