"""Vietnamese scientific report; every quantitative claim comes from real artifacts."""
from raw_temporal_common import *
from report_raw_temporal_audit import md
T=OUT/'tables'
def read(name):return pd.read_csv(T/(name+'.csv'))
def main():
    assert json.loads((OUT/'xai/summary_complete.json').read_text())['complete']
    primary=read('primary_ablation').set_index('model');reference=primary.loc['kinematic_lightgbm_reference'];nested=primary.loc['nested_selected'];hier=primary.loc['hierarchical']
    mainnames=['train_majority_reference','kinematic_lightgbm_reference','cnn','tcn','transformer','hierarchical','nested_selected'];cols=['f1_macro','accuracy','auc','balanced_accuracy','sensitivity','specificity']
    foldtable=[]
    for name in mainnames:
        row={'model':name}
        for col in cols:row[col]=f"{primary.loc[name,'mean_'+col]:.4f} ± {primary.loc[name,'std_'+col]:.4f}"
        foldtable.append(row)
    timeab=read('temporal_multiscale_ablation');seeds=read('seed_sensitivity');cost=primary.loc[['cnn','tcn','transformer','hierarchical','no_explicit_time','no_position','index_time','local_stroke'],['parameters','parameters_max','train_seconds_mean','inference_seconds_mean','peak_gpu_bytes']].reset_index();cost['peak_tensor_MiB']=cost.peak_gpu_bytes/1024**2
    tasks=read('task_importance');top=tasks[(tasks.model=='hierarchical')&(tasks.seed==42)].sort_values('rank');foldst=read('fold_xai_stability').groupby('model')[['spearman','top3_jaccard']].agg(['mean','std']).reset_index();foldst.columns=['_'.join(filter(None,c))if isinstance(c,tuple)else c for c in foldst.columns]
    seedst=read('seed_xai_stability').groupby(['model','fold']).spearman.mean().groupby('model').agg(['mean','std']).reset_index()
    geometry=read('representation_geometry_stability').groupby(['model','fold']).spearman.mean().groupby('model').agg(['mean','std']).reset_index()
    over=read('overfitting').groupby('model')[['best_epoch','loss_gap','max_epochs_reached']].mean().reset_index();perturb=read('temporal_perturbation_summary');pair=read('paired_comparisons');ig=read('ig_validation');faith=read('faithfulness');sanity=read('ig_baseline_seed_randomization');cases=read('selected_cases');audit=json.loads((OUT/'audit/summary.json').read_text());prepared=json.loads((OUT/'cache/validation.json').read_text());prov=json.loads((OUT/'provenance.json').read_text())
    decision_summary=read('decision_diagnostics').query('seed==42').groupby('model').agg(single_class_folds=('single_class_predictions','sum'),folds=('fold','size'),predicted_positive=('predicted_positive','sum')).reset_index()
    old_tasks=pd.read_csv(ROOT/'results/kinematic_xai/exploratory/tables/task_importance.csv').sort_values('global_rank')
    old_groups=pd.read_csv(ROOT/'results/kinematic_xai/exploratory/tables/group_importance.csv').sort_values('global_rank')
    best=primary.loc[['cnn','tcn','transformer','hierarchical']].f1_macro.idxmax()
    rq2=primary.loc['transformer','f1_macro']-max(primary.loc['cnn','f1_macro'],primary.loc['tcn','f1_macro'])
    rq3=primary.loc['hierarchical','f1_macro']-primary.loc['transformer','f1_macro']
    case_figures='\n\n'.join(f'''### Ca {r.category}: {r.participant_id}

![Tín hiệu gốc và temporal attribution](figures/case_{r.category}_raw_signals.png)

[Trajectory và tín hiệu SVG](figures/case_{r.category}_raw_signals.svg) · [Stroke effects](figures/case_{r.category}_stroke_effects.svg) · [Khoảng thời gian phóng to](figures/case_{r.category}_temporal_zoom.svg) · [Phân rã IG](figures/case_{r.category}_ig_components.svg)'''for r in cases.itertuples())
    txt=f'''# Raw Temporal Graphomotor Representation Learning — báo cáo Task 2 thay thế

## Kết quả chính và phạm vi kết luận

Đây là nghiên cứu **raw chronological signals**, thay thế hoàn toàn thí nghiệm cũ trên handcrafted task vectors. Code/results cũ đã xóa khỏi active branch bằng normal Git commit; lịch sử public không bị viết lại. Cache/checkpoints cũ được giữ local dưới `data/derived/retired_task2_025b840/`, không tham gia benchmark. Không tái đưa mô hình cũ vào bảng so sánh.

Protected Kinematic + LightGBM đạt Accuracy OOF **{reference.accuracy:.4%}**, macro-F1 **{reference.f1_macro:.4f}**, ROC-AUC **{reference.auc:.4f}**. Quy trình chọn raw architecture bằng inner CV (`nested_selected`) đạt Accuracy **{nested.accuracy:.4%}**, macro-F1 **{nested.f1_macro:.4f}**, ROC-AUC **{nested.auc:.4f}**; Δmacro-F1 OOF so với reference **{nested.f1_macro-reference.f1_macro:+.4f}**.

{'Chưa có bằng chứng để thay thế baseline Kinematic bằng raw temporal model trên cohort này.'if nested.f1_macro<=reference.f1_macro else 'Raw nested selection có điểm OOF cao hơn trên cohort này; đây vẫn là exploratory reuse, chưa chứng minh vượt trội ngoài cohort hoặc lâm sàng.'}

{md(primary.loc[mainnames,cols].reset_index())}

Các bảng metric dùng thang0–1; Accuracy viết bằng phần trăm trong đoạn văn được ghi dấu%. Sensitivity/specificity lần lượt xem group1/group0 là lớp cần phát hiện, không đổi chúng thành nhãn chẩn đoán lâm sàng.

Fold mean±SD (ddof=1), khác pooled OOF:

{md(pd.DataFrame(foldtable))}

Không chọn mô hình tốt nhất sau khi xem outer Test rồi gọi score của nó là unbiased. `{best}` có macro-F1 OOF mô tả cao nhất trong bốn raw families; đó là **xếp hạng hậu nghiệm**, không thay cho nested selector đã khai báo. Tất cả conclusions vẫn exploratory vì cùng cohort/outer folds đã được nghiên cứu trước đó; cần external validation.

## 1. Động cơ nghiên cứu và mối liên hệ Task 1

Kinematic features đã chứa thông tin temporal như velocity, acceleration, jerk, duration và FFT. Vấn đề là aggregation có thể mất thứ tự sự kiện, động học cục bộ, pauses/resumptions và tương tác giữa strokes; không nói phương pháp trước “không có temporal information”. Giả thuyết là học trực tiếp từ raw progression có thể giữ thêm cấu trúc có ích. Không giả định Transformer tốt hơn CNN/TCN hoặc LightGBM, không giả định language invariance hay biological validity.

Task-1 TreeSHAP cho thấy copy/dictation và các feature groups có vai trò trong mô hình summary; điều đó chỉ gợi ý kiểm tra động học bên trong task. Không dùng SHAP cũ làm neural weights, không Top-K task selection. [Task 1 giữ nguyên](../kinematic_xai/REPORT_VI.md). Các nhánh GAF/CWT/MOMENT, fusion, Knowledge Graph, domain adversarial và cross-language không được triển khai trong nghiên cứu này.

Năm task và năm nhóm đứng đầu trong artifact Task1 đã kiểm chứng:

{md(old_tasks[['task','n_features','share_percent','mean_rank','rank_sd']].head(5))}

{md(old_groups[['group','n_features','share_percent','mean_rank','rank_sd']].head(5))}

`share_percent` là tỷ trọng tổng absolute feature-SHAP mass trong mô hình Kinematic, không phải phần trăm xác suất hoặc grouped Shapley value. Các nhóm có số features khác nhau; các bảng này không xác nhận cơ chế sinh học. FFT và trajectory trong summary đã chứa temporal information; raw temporal kiểm tra thêm cấu trúc tiến triển chưa được giữ bởi aggregation.

## 2. Data audit và timestamp anomalies

Giữ 258 participants, 14 tasks và3.612 recordings với **{audit['samples']:,} raw samples**, nhãn `group` (115/143), đúng original fold assignments seed42. Không đổi sang diagnosis flag. Group là grouping trong dataset, không đồng nhất hoàn toàn chẩn đoán dysgraphia. Channels được xác minh theo tên: x,y,time,pen_status,pressure,azimuth,tilt; source schema order cũng được giữ trong audit.

Có {audit['duplicate_dt']:,} duplicate intervals, {audit['negative_dt']} negative intervals, {audit['large_gaps']:,} gaps>1s, không có nonfinite/missing channel trong cohort. Median samples/recording={audit['n_samples_quantiles']['0.5']:.0f}, maximum={audit['n_samples_quantiles']['1.0']:.0f}. Không mặc định sampling uniform. Effective sampling rate toàn duration và1/median positive Δt được báo cáo riêng.

[Audit tiếng Việt đầy đủ](audit/REPORT_VI.md), `audit/recordings.csv`, `pen_state_runs.csv` và `source_hashes.json` chứa số liệu cho mọi recording. Time dùng source seconds convention như baseline, chưa có calibration certificate; coordinate/pressure giữ đơn vị thiết bị, không tự đổi thành mm hoặc Newton. Không có chuyên gia gán ground-truth tremor/stroke/biomarker.

## 3. Preprocessing và alignment

Không sort timestamps, không drop duplicate, không resample cả task sang cùng số điểm. Giữ acquisition order và raw index. Δt≤0, Δt<0,0001 hoặc>1 giây tạo derivative/segment boundary; giữ timestamps và flags. Motion derivatives chỉ tính khi actual Δt hợp lệ và cùng pen-down run; acceleration cần hai velocity intervals liên tiếp hợp lệ. Invalid derivative đặt0 cùng validity mask0, không coi0 là quan sát vận động thật. Không tính derivative xuyên qua pen-up. Không dùng jerk ở phiên bản đầu vì độ nhạy đạo hàm bậc cao.

21 sample channels gồm relative x/y, pressure, orientation values, actual elapsed/Δt, pen state, vx/vy/speed, ax/ay và validity/anomaly/boundary indicators. Signed log1p giảm dynamic range. Per-record origin subtraction loại page offset nhưng giữ kích thước và thời lượng; có thể bỏ thông tin absolute placement hữu ích. Azimuth/tilt giữ số gốc, chưa unwrap/calibrate góc; wrap discontinuities là một hạn chế.

Channel normalization **chỉ fit trên current Train participants**: gộp first/second moments với mỗi participant có trọng số bằng nhau; fixed SD floor0,1, z saturation±10. Có thể làm mất tín hiệu biên độ cực đoan, không gọi mọi extreme là nuisance. Không dùng precomputed74-dimensional Kinematic vectors làm neural input. Cohort IDs/labels chỉ lấy từ validated cache để giữ protocol.

Cache deterministic lưu float32; packed float16 storage chia sẻ qua memory map có maximum input log-channel quantization error0,00390625, được kiểm tra trên toàn dữ liệu. Model/gradient float32, không dùng AMP. Original raw data và inventory timestamps vẫn float64; không thay normalized time axis cho physical duration.

Mapping bắt buộc: participant→task source key→raw index/time/pen-state run→anomaly-split segment→128-sample window. Mapping cho XAI lưu cả original run index (gồm pen-up) và down-stroke index. Không coi every pen-down run là một nét chữ đã được chuyên gia xác nhận.

## 4. Segmentation và full coverage

{int(pd.read_csv(OUT/'audit/prepared_inventory.csv').patches.sum()):,} non-overlapping windows, tối đa128 samples, không vượt pen-state/anomaly boundary; long strokes chia nhỏ, short windows padding có mask. **{audit['samples']:,}/{audit['samples']:,}** samples được phủ đúng một lần. Không truncate hoặc random crop recording. Physical duration của windows khác nhau. So sánh inventory64/128/256 trong audit là trade-off tính toán, không phải predictive window-size tuning.

CNN kernel9 stride8 có right-edge padding bổ sung và output-validity rule dựa trên receptive-field overlap; impulse test kiểm tra mọi vị trí thật, bao gồm phần cuối stride. Một lỗi edge-coverage được phát hiện ở kiểm thử phát triển, đã archive toàn bộ partial inner fits và chạy lại từ đầu; không có outer Test predictions của phiên bản lỗi được dùng. Archive nằm trong ignored `data/derived/raw_temporal_debug_pre_coverage_fix/`. Không phục hồi abandoned handcrafted experiment.

## 5. Bốn raw architectures

| ID | Model | Cấu trúc |
|---|---|---|
| R1 | CNN | Local convolutional movement encoder, bounded patch pooling, task-ID feed-forward aggregation |
| R2 | TCN | Cùng local encoder, ba dilated convolution blocks trên patch tokens trong bounded task chunks |
| R3 | Temporal Transformer | Cùng local encoder, continuous-time encoding, self-attention trên patch tokens trong task chunks |
| R4 | Hierarchical | Local CNN→stroke attention→task temporal attention→task-aware participant attention |

CNN dùng Conv1d21→8, kernel9/stride8, Conv1d8→32 kernel3 và masked pooling. TCN dùng dilation1/2/4, không tuyên bố causal forecasting (padding hai phía, đây là phân loại sau recording). Transformers d32,2heads, một block ở mỗi mức tương ứng, residual/LayerNorm và dropout0,1 hoặc0,25. Continuous-time MLP nhận elapsed source time, window/segment duration, mean physical interval và sample count sau signed-log scaling. Đây là thời gian thực trong recordings;14 task IDs **không có chronological positional encoding**.

R4 attention bounded16 windows/stroke chunk và32 segments/task chunk; mọi chunk được pool, không bỏ cuối chuỗi. R3 và TCN dùng32 patch tokens/chunk. Không có global quadratic attention trên recording200.000 points. Pooling cho phép tóm tắt spans dài nhưng không trực tiếp attention giữa mọi distant sample; receptive field/chunking là giới hạn cần công khai.

Các kiến trúc khác nhau về context và capacity: R1 khoảng4k parameters, R2/R3 khoảng13k, R4 khoảng30k; exact counts trong bảng chi phí. **Chưa capacity-match hoàn toàn**, do đó chênh lệch không thể gán riêng cho attention hoặc hierarchy theo nghĩa nhân quả. Primary inputs/splits/budgets/selection/seeds/threshold được giữ thống nhất.

![Kiến trúc](figures/03_raw_temporal_architecture.png)

## 6. Nested participant-independent evaluation

Outer6-fold cố định, mỗi Test43 người, Train215; inner Stratified3-fold seed42. Mọi windows/tasks/strokes của participant nằm cùng partition. Independent sample count là258, không phải{int(pd.read_csv(OUT/'audit/prepared_inventory.csv').patches.sum()):,} windows. Candidate search chỉ hai mức dropout cho mỗi primary family; không có grid lớn hoặc chọn window size từ Test.

AdamW lr0,001, weight decay0,001, gradient clip1; micro-batch8 participants, effective batch8; maximum20epochs, patience4, minimum6. Chọn inner checkpoint theo validation BCE cải thiện ít nhất1e-4. Chọn candidate theo mean inner macro-F1, tie-break AUC, candidate index; architecture winner cũng chọn chỉ từ inner scores. Final epochs là median best inner epoch và fit toàn outer Train, không monitor Test trong fit. Threshold probability>0,5 cố định.

`minimum_epochs=6` là số epochs tối thiểu trước khi cho phép dừng inner run; best checkpoint có thể nằm ở epoch1–5 và median epoch dùng cho final fit có thể nhỏ hơn6. Các history JSON final chỉ có Train metrics. Dòng console `val0.0` của final fit là giá trị mặc định khi không có validation, không phải loss đo trên outer Test.

216 inner fits và96 final fits đã được thực hiện. Primary4 families lặp seeds42/43/44 với configuration chọn bằng seed42; bốn matched ablations dùng seed42. Seeds không phải independent cohorts; không gộp18 folds như18 quan sát độc lập. Baseline LightGBM dùng original OOF/checkpoints, không retrain hoặc optimize trong nghiên cứu này.

![So sánh dự đoán](figures/04_predictive_comparison.png)

## 7. Temporal-information và multi-scale ablation

OOF của các ablations:

{md(primary.loc[['hierarchical','no_explicit_time','no_position','index_time','local_stroke'],cols].reset_index())}

Paired fold Δ so với full R4:

{md(timeab)}

`no_explicit_time`: tắt time-embedding và explicit elapsed/Δt/anomaly channels; **vẫn còn velocity/acceleration tính bằng actual Δt và integrity boundaries**, nên không được gọi là hoàn toàn time-free. `no_position`: tắt temporal embedding, nhưng CNN vẫn có local order và sample channels vẫn chứa clocks; không phải phép loại toàn bộ order. `index_time`: dùng sample-order encoding và ordinal motion differences, không dùng physical elapsed/Δt hoặc clock flags, nhưng vẫn giữ boundary inventory để không làm hỏng dữ liệu; còn residual structural timing information.

`local_stroke`: giữ local+stroke modeling, bỏ task temporal và participant attention; đối chiếu với R1 local-only và full R4 cho multi-scale. Hyperparameter/dropout của matched variants lấy từ R4 inner selection, rồi chọn epoch bằng inner validation riêng. Không chọn ablation theo outer Test. Capacity/context và feature representations khác nhau như đã nêu, không diễn giải mọi Δ là tác động đơn độc một yếu tố.

Stroke-order controls đảo/hoán vị **complete pen-state runs**, giữ nội bộ các segments/stroke. Segment slot timing và clock information trong embeddings có thể mâu thuẫn; đó là OOD sensitivity test, không phải trajectory vật lý bình thường. Missing segments dùng boolean masks/key removal, không chèn zero-valued movement vào chuỗi.

{md(perturb)}

Reverse repeats giống nhau theo thiết kế, không xem là ba phép thử độc lập. Random orders/masks dùng seeds cố định; repeats được trung bình trong fold trước khi mô tả mean±SD sáu folds.

Trong kết quả này, `no_position` có macro-F1 OOF {primary.loc['no_position','f1_macro']:.4f}, so với full R4 {hier.f1_macro:.4f} (Δ={primary.loc['no_position','f1_macro']-hier.f1_macro:+.4f}). Đảo run làm thay đổi xác suất tuyệt đối trung bình {perturb.set_index('kind').loc['reverse_runs','mean_absolute_probability_shift_mean']:.6f}, với Δmacro-F1 fold trung bình {perturb.set_index('kind').loc['reverse_runs','delta_f1_macro_mean']:+.6f}. Đây chưa phải bằng chứng cơ chế temporal ordering mang lại lợi ích: mô hình gốc đã dự đoán yếu, có thể ít nhạy với perturbation vì các đường pooling/metadata còn lại. Không suy từ độ bền này thành robustness lâm sàng hoặc bất biến thứ tự nói chung.

## 8. Overfitting, seeds và chi phí

Kiểm tra quyết định ở ngưỡng0,5; một fold dự đoán duy nhất một lớp vẫn có thể có ROC-AUC khác0,5 nếu xác suất còn biến thiên. Đối chứng majority lấy lớp phổ biến chỉ từ outer Train; score0/1 là hard reference, không phải xác suất đã calibration. Không đổi threshold sau khi xem bảng này.

{md(decision_summary)}

R4 seed42 có {int(decision_summary.set_index('model').loc['hierarchical','single_class_folds'])}/6 folds dự đoán một lớp. Khi hiện tượng này xảy ra, đó là hạn chế của decision rule trong budget này; không đồng nghĩa latent representations hoặc xác suất hoàn toàn hằng. BCE early stopping, class prior và calibration có thể liên quan, nhưng bảng hiện tại không xác định riêng nguyên nhân. Không đổi loss, threshold hay epoch sau khi quan sát outer Test để làm đẹp kết quả.

{md(over)}

Train curves là online loss với dropout và weights đang cập nhật; inner-validation chạy eval. Loss gap là chẩn đoán mô tả, không phải unbiased overfitting estimator. `max_epochs_reached` thể hiện tỷ lệ inner runs chạm budget20; nếu lớn, chưa thể gọi search là hội tụ tối ưu. Early stopping có thể chọn epoch rất sớm trên validation nhỏ; ngân sách cố định hạn chế khả năng kết luận kiến trúc nào tối ưu tuyệt đối.

{md(seeds)}

{md(cost)}

Chi phí của toàn bộ312 lượt fit, tách tìm cấu hình trong inner CV khỏi fit cuối (gồm ba seed cho các mô hình chính):

{md(read('computational_cost_by_scope'))}

`train_seconds_sum` chỉ cộng thời gian các fit đã hoàn tất và được lưu; không phải wall-clock từ lúc bắt đầu đến lúc kết thúc phiên làm việc. Không cộng thời gian chờ, phần fit bị ngắt, source audit, cache preparation hoặc XAI. `evaluation_seconds_sum` là forward đánh giá sau fit; inner fit đã chứa các lượt validation trong quá trình early stopping.

Inference column là thời gian toàn43 held-out participants trên GPU, gồm cached-input normalization/transfer/forward, không gồm raw extraction. `tables/latency.csv` đo riêng warm CPU forward từng participant từ normalized tensors, với hai CPU threads; hai đại lượng khác thiết bị và scope, không được dùng để suy ra GPU speedup. XAI chạy CPU để không tranh VRAM với training. Train time inner có validation overhead; final fit không có validation. Peak là allocated tensor memory của process, không gồm toàn driver/context/reserved memory. Benchmark cuối cùng chạy tuần tự một GPU worker trên Quadro T1000 4GB; không trộn các partial fits trước khi khóa implementation/runtime.

XAI CPU có thể chạy song song với worker GPU và cùng dùng tài nguyên host; vì vậy thời gian Train/inference/latency là chi phí quan sát trong workflow này, không phải dedicated hardware benchmark. Không diễn giải chênh lệch nhỏ thành lợi thế tốc độ chắc chắn của kiến trúc.

Mỗi participant được forward một lần không tính giờ để warmup đúng ragged shape, sau đó đo một forward CPU. Độ phân tán latency vì vậy chứa khác biệt độ dài recording và host contention, không phải sai số của nhiều lần đo cùng một input. Hai fold XAI trước khi sửa warmup được archive; bảng equivalence kiểm tra rằng mọi output giải thích/embedding được giữ nguyên, chỉ số đo thời gian thay đổi.

Host CPU: {prov['cpu_hardware']['Name']} ({prov['cpu_hardware']['NumberOfCores']} cores/{prov['cpu_hardware']['NumberOfLogicalProcessors']} logical processors). Training dùng bốn CPU threads, XAI/CPU latency dùng hai threads. GPU Quadro T1000 có4.096MiB VRAM, driver {prov['gpu_driver']}; package versions lưu trong `provenance.json`.

Pilot trên tám participants có số samples lớn nhất trong outer Train0 đạt peak tensor {prov['largest_outer_train_microbatch_peak_tensor_bytes']/1024**2:.1f}MiB; không dùng Test cho resource/model tuning. Các thử nghiệm concurrency ban đầu được ghi trong provenance; benchmark cuối dùng một worker với micro-batch8. TCN dùng unfold+GEMM tương đương dilated convolution và có test output/input-gradient/weight-gradient. Deterministic preprocessing SVC mất {prepared['seconds']:.1f}s, audit mất {audit['seconds']:.1f}s; packing/statistics là chi phí một lần, không cộng vào forward latency. Baseline training cost để N/A vì không retrain baseline.

## 9. Paired failure analysis

{md(pair[pair.model.isin(mainnames)])}

Corrected/harmed dùng cùng participants so với protected baseline; không chọn threshold lại. `participant_changes.csv` giữ probabilities/outcomes cho mọi người, không chỉ ca thuận lợi. Different error cases gợi ý giả thuyết bổ sung, **chưa chứng minh conditional complementary information** hay fusion gain; fusion chưa được chạy.

## 10. Hierarchical XAI và temporal alignment

Task importance: missing-task occlusion cho toàn258 người, bốn raw models và ba seeds. Importance là absolute change của positive-class logit hoặc probability, không phải grouped Shapley value và không phải clinical relevance. Tính thêm fold macro-F1 drop khi thiếu task để kiểm chứng ranking theo criterion khác.

Task occlusion đồng thời loại nội dung, task identity và cấu trúc hiện diện của tác vụ. Hiệu ứng vì vậy có thể phản ánh prior của mô hình về tác vụ, không chỉ chuyển động. Integrated Gradients giữ task IDs, segmentation và masks cố định; attribution là lời giải thích có điều kiện trên cấu trúc đó. Không suy ra số tác vụ tối thiểu hoặc ý nghĩa chẩn đoán từ hai phép phân tích này.

{md(top)}

![Task-level occlusion](figures/09_task_importance.png)

Stroke/segment importance: R4 seed42 có joint occlusion cho mọi original pen-state run và từng safe segment của tất cả held-out participants. Giữ original run/down-stroke indices, raw start/end, physical timestamps, pen state và dấu logit effect. Temporal occlusion chia mỗi task thành tối đa8 contiguous patch blocks, che bằng masks, giữ mapping cho mọi interval. Task occlusion không phải tổng các stroke effects vì mô hình nonlinear.

Integrated Gradients cho bộ ca minh họa cố định giải thích **positive-class logit**, không phải probability. Tích phân đồng thời sample channels, patch-time metadata và segment-time metadata; segmentation inventory/masks giữ cố định. Hai references: `zero_joint` và `structure_preserving` (giữ pen/validity/boundary flags và count metadata). Đây là mathematical references, không phải recording khỏe mạnh có thật. Tổng IG được kiểm tra bằng logit(x)−logit(reference), tolerance0,003+0,003×|delta|; tăng Gauss–Legendre steps16→256 để hội tụ, không nới tolerance theo Test.

{md(ig[['participant_id','category','seed','baseline','kind','steps','residual']])}

Dense IG lưu local trong `xai/examples/*.npz`. Sample-channel IG ánh xạ về đúng raw samples; metadata attributions lưu riêng theo window/segment. Không tự rải metadata attribution lên raw points rồi gọi đó là pointwise ground truth. Temporal overlays chỉ hiển thị signed sample-channel contributions và ghi rõ phần đó.

Reference được định nghĩa trong không gian input **đã normalization/log-transform**: continuous channels bằng0 thường tương ứng tâm Train của log-channel, không phải mọi đại lượng vật lý bằng0. Sample-channel IG cũng bao gồm elapsed/Δt và flags, không chỉ x/y hoặc các đạo hàm vận động. Inventory và độ dài thật vẫn giữ cố định; do đó đường tích phân không nhất thiết là chuỗi chuyển động vật lý hợp lệ.

IG xem21 input channels như các tọa độ của representation đã chuẩn bị, dù velocity/acceleration phụ thuộc x/y/time trong dữ liệu thật. Chưa áp dụng chain rule xuyên qua toàn bộ preprocessing và việc tạo masks để đưa attribution về bảy raw signals độc lập. Overlay theo raw index/time vì vậy giải thích reliance trên representation tại các thời điểm đó, không phải hiệu ứng vật lý hay nhân quả của từng sensor channel.

`tables/ig_components.csv` và hình `case_*_ig_components` tách reference logit, tổng signed sample-channel IG và hai tổng time-metadata IG. Phân rã này cho thấy overlay tín hiệu chỉ là một phần của output; reference vẫn giữ topology/task identity theo định nghĩa phép giải thích. Không chuyển các tổng logit này thành phần trăm xác suất.

## 11. Stability, faithfulness và sanity checks

Task rank stability qua folds:

{md(foldst)}

Với R4, Spearman trung bình giữa các cặp fold là {foldst.set_index('model').loc['hierarchical','spearman_mean']:.4f} và top-3 Jaccard là {foldst.set_index('model').loc['hierarchical','top3_jaccard_mean']:.4f}. Thứ hạng task hiện không ổn định qua folds; danh sách task đứng đầu khi gộp OOF chỉ nên dùng mô tả mô hình này, chưa đủ để đề xuất rút gọn bộ đánh giá. Các cặp fold phụ thuộc nhau, không phải các lần xác nhận độc lập.

Task rank stability qua seeds (trung bình participants trong fold rồi mô tả sáu folds):

{md(seedst)}

Representation geometry qua seeds, dùng Spearman của pairwise Euclidean distances trên cùng43 Test participants, bất biến rotation/translation/global scale:

{md(geometry)}

![Độ ổn định task importance và hình học embedding](figures/10_explanation_and_geometry_stability.png)

903 distances/fold không phải903 participants độc lập. Không đối chiếu trực tiếp từng latent coordinate giữa hai seeds hoặc hai folds vì không gian có thể quay/dịch. Cross-fold task ranks dùng held-out subjects khác nhau; không giải thích chênh lệch hoàn toàn là instability của một người.

IG so với independent temporal/task occlusion trên ca minh họa:

Task IG mass gồm absolute attribution của cả sample channels, patch metadata và segment metadata. Temporal-block correlation và IG-top/bottom window deletion chỉ dùng sample-channel + patch-metadata mass; không phân bổ segment-metadata IG tùy ý xuống các windows. Vì vậy đây là kiểm chứng một phần representation, không phải so sánh toàn bộ joint IG với một ground-truth temporal explanation.

{md(faith)}

Trên toàn cohort, tương quan giữa task importance theo mean absolute logit change và task-level macro-F1 drop của R4 là {read('task_faithfulness').set_index('model').loc['hierarchical','spearman']:.4f}. Độ lớn thay đổi output vì vậy không tự là thước đo thứ hạng giá trị dự đoán của task; tương tác giữa tasks và ngưỡng quyết định có thể làm hai đại lượng khác nhau. Không dùng bảng importance để kết luận số tác vụ tối thiểu.

Sensitivity với baseline, seeds và model randomization:

{md(sanity)}

Trong bốn ca đã chọn, patch-level rank correlation trung bình sau randomization toàn mô hình là {sanity[sanity.comparison=='random_all'].patch_spearman.mean():.4f}; sau randomization head là {sanity[sanity.comparison=='random_head'].patch_spearman.mean():.4f}. Attribution vẫn giữ tương quan đáng kể với mô hình gốc ở mức patch, nên sanity check không hỗ trợ diễn giải các vị trí nổi bật như bằng chứng riêng của trọng số đã học. Cấu trúc đầu vào, độ dài và masks có thể góp phần nhưng chưa được tách bằng thí nghiệm riêng. Đây là giới hạn thực tế của explanations hiện tại, không được che bằng việc IG đạt completeness.

![Faithfulness và randomization sanity checks](figures/12_ig_faithfulness_and_sanity.png)

Random head/full-model reinitialization chỉ phục vụ sanity checks, không là kiến trúc mới hoặc candidate để tối ưu accuracy. Correlation còn cao sau randomization nếu có là giới hạn đáng báo cáo, không ép về0. Completeness chứng minh phép tích phân khớp output, **không chứng minh explanation faithful theo nghĩa nhân quả hay sinh học**. Occlusion có thể OOD hoặc tương tác feature redundancy. IG-top/bottom deletion curves lưu riêng, chỉ trên documented examples, không suy thành test lâm sàng.

Các bảng segment/run dùng sigmoid trên scalar float32 từ encoder, còn temporal-block output được chuyển sang float64 trước sigmoid. Validator tái tạo đúng phép tính từng bảng với sai số đối chiếu1e-12, đồng thời kiểm tra chênh lệch với phép tính sigmoid float64 không quá một epsilon float32; giá trị lớn nhất lưu trong `xai_validation.json`. Đây là roundoff của xác suất, không phải khác biệt checkpoint hay thay đổi attribution.

Task ranking so với existing Kinematic TreeSHAP được giữ trong `kinematic_xai_rank_comparison.csv`, chỉ descriptive vì phương pháp/reference khác nhau; không dùng làm training weights hoặc ép nhất quán.

## 12. Ca TP/TN/FP/FN và hình thực

{md(cases)}

Các khoảng phóng to được mô tả bằng số liệu trực tiếp từ SVC gốc:

{md(read('behavioral_examples')[['participant_id','category','task','raw_start','raw_end_exclusive','samples','duration','pen_state']])}

{md(read('behavioral_examples')[['participant_id','valid_speed_samples','speed_median','speed_max','pressure_median','acceleration_median','invalid_time_samples']])}

Speed/acceleration dùng coordinate units/source second và coordinate units/source second²; pressure giữ source units. NaN nghĩa là đoạn không có đủ intervals hợp lệ để tính đạo hàm, không thay bằng0 hoặc kết luận người viết đứng yên. Pen-up không tự đồng nghĩa một pause có ý nghĩa bệnh lý. Đây là mô tả các đoạn theo rule đã định, không phải đặc trưng được chọn lại để huấn luyện.

Chọn participant ID nhỏ nhất theo từ điển trong mỗi outcome của R4 seed42 OOF. Nếu một outcome không xuất hiện, không tạo ca giả hoặc đổi threshold để có đủ. Cùng bộ ca giữ cố định cho mọi seed/reference/randomization. Task hiển thị là task có |task-occlusion logit effect| lớn nhất ở participant đó, rule được công khai.

Hình dùng trajectory thật tô màu theo source elapsed time, speed/acceleration chỉ ở valid intervals, pressure, pen state, boundaries, temporal IG và run-level effects. Dense traces được rasterize khi nhúng trong SVG để giảm kích thước, không resample dữ liệu phân tích. Attribution spike không chứng minh tremor, motor impairment hoặc dysgraphia. Attention chỉ là supplementary, không dùng thay độc lập IG/occlusion.

{case_figures}

## 13. Trả lời câu hỏi nghiên cứu

**RQ1:** Raw nested selection đạt F1={nested.f1_macro:.4f} so với Kinematic {reference.f1_macro:.4f}. {'Chưa vượt reference.'if nested.f1_macro<=reference.f1_macro else 'Có mức điểm mô tả cao hơn reference trong cohort này.'} Kể cả có corrected cases, chưa chứng minh thông tin bổ sung có điều kiện ngoài summaries; việc đó cần nghiên cứu độc lập/fusion trong tương lai, chưa thực hiện ở đây.

**RQ2:** Flat Temporal Transformer trừ raw CNN/TCN tốt hơn theo OOF: ΔF1={rq2:+.4f}. {'Không thấy lợi ích dự đoán từ flat self-attention so với baseline convolution tốt hơn trong protocol này.'if rq2<=0 else 'Có tín hiệu mô tả có lợi cho flat attention, nhưng phải xét fold/seed spread và capacity khác nhau.'} Không chọn outer winner rồi gọi là xác nhận độc lập.

**RQ3:** R4 trừ R3 ΔF1={rq3:+.4f}; multi-scale controls và parameter counts được báo cáo song song. {'Chưa có bằng chứng hierarchy giúp hơn flat temporal attention trên OOF này.'if rq3<=0 else 'Hierarchy có điểm mô tả cao hơn flat model; chưa tách hoàn toàn hiệu ứng hierarchy khỏi capacity/context.'}

**RQ4:** Bảng trained time ablations và held-out run ordering/missing-segment controls định lượng reliance. Không gộp `no_position`, `no_explicit_time` và `index_time` thành cùng một khái niệm; chúng giữ/loại thông tin khác nhau. Performance thay đổi dưới OOD perturbation chưa là proof về clinically meaningful temporal learning.

**RQ5:** Có explanations thật ở task/run/segment/raw-sample levels, completeness, alignment, occlusion, two references và randomization. Mức agreement/sanity trong các bảng thể hiện giới hạn; không coi attention hoặc IG completeness tự đủ để bảo đảm faithful clinical interpretation. IG ví dụ không đại diện toàn bộ cohort, còn coarse occlusion được chạy trên mọi held-out participant.

**RQ6:** Seed predictive metrics, task importance và representation-distance stability đã đo riêng. Không gian latent không cần giữ cùng tọa độ; uncertainty giữa folds/subjects và repeated seeds không độc lập. Stability cao nếu có không bù được accuracy thấp, và chưa có external/cross-language validation.

## 14. Hạn chế và bước nghiên cứu tiếp theo

258 independent subjects vẫn nhỏ so với54,8 triệu correlated samples. Có adaptive research bias do từng xem các outer folds; có sensor/time anomalies và thiếu calibration/clinical annotation. Bounded context, log scaling/saturation/half storage, unequal model capacity và fixed20-epoch budget có thể hạn chế kết quả. Orientation chưa được unwrap/calibrate. Missing masks và order perturbations không luôn là biến đổi label-preserving hoặc recording thực tế.

Không tối ưu lại baseline, không đưa handcrafted features vào neural input, không fusion, không Top-K XAI, không cross-language. Hướng tiếp theo sau review có thể kiểm chứng calibration/segmentation và external cohort; nếu nghiên cứu complementarity/fusion cần protocol riêng, không suy từ overlap errors hiện tại. Không tự triển khai các bước đó.

## 15. Deliverables và tái lập

Code ở `src/diagramo/temporal/`, scripts tên `*_raw_temporal*.py`, config `configs/raw_temporal.json`; outputs duy nhất của nghiên cứu mới ở `results/raw_temporal/`. [Notebook bảng số](../../notebooks/06_raw_temporal_graphomotor.ipynb) đọc artifacts, không train khi mở. Figures PNG300dpi/SVG có source/output hashes và captions; paper drafts chưa xác minh quy định journal cụ thể.

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/audit_raw_temporal_sources.py
.venv/Scripts/python.exe -X utf8 -B scripts/audit_raw_temporal.py
.venv/Scripts/python.exe -X utf8 -B scripts/prepare_raw_temporal.py
.venv/Scripts/python.exe -X utf8 -B scripts/prepare_raw_temporal_statistics.py
.venv/Scripts/python.exe -X utf8 -B scripts/pack_raw_temporal.py
.venv/Scripts/python.exe -X utf8 -B scripts/train_raw_temporal.py
.venv/Scripts/python.exe -X utf8 -B scripts/evaluate_raw_temporal.py
.venv/Scripts/python.exe -X utf8 -B scripts/analyze_raw_temporal_xai.py --device cpu
.venv/Scripts/python.exe -X utf8 -B scripts/analyze_raw_temporal_xai.py --examples-only --device cpu
.venv/Scripts/python.exe -X utf8 -B scripts/summarize_raw_temporal_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/validate_raw_temporal.py --phase data
.venv/Scripts/python.exe -X utf8 -B scripts/validate_raw_temporal.py --phase fits
.venv/Scripts/python.exe -X utf8 -B scripts/validate_raw_temporal.py --phase xai
.venv/Scripts/python.exe -X utf8 -B scripts/report_raw_temporal.py
.venv/Scripts/python.exe -X utf8 -B scripts/plot_raw_temporal.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_raw_temporal_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 06_raw_temporal_graphomotor.ipynb --validation-output results/raw_temporal/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_raw_temporal.py --phase final
```

Checkpoint/deterministic caches/dense IG giữ local, không thay thế raw recordings. Training signature ngăn resume với code/config khác. Data validation đối chiếu mọi cached point với raw preprocessing; fit validation kiểm tra IDs/normalizers/inner choices/epochs và replay mọi inner/outer prediction; XAI validation kiểm tra toàn bộ mapping/arithmetic và replay task occlusion cùng first/last segments mỗi task, không giả vờ đã replay mọi dense occlusion. Reproducibility manifest giữ code/artifact hashes.

Chỉ coi nghiên cứu hoàn tất khi `validation.json` có **complete=true** sau các kiểm tra, notebook execution và16 raw-temporal +8 Task1 +6 baseline regression tests. Không push/merge hay viết lại remote history trong nhiệm vụ này.
'''
    source=json.loads((OUT/'audit/raw_svc_comparison.json').read_text())
    source_note=f'''## Nguồn SVC gốc và xuất dữ liệu merged

Neural input đọc trực tiếp toàn bộ3.612 file `.svc` gốc, có **{audit['samples']:,} samples**. Merged JSON chỉ cung cấp mapping task/cohort và metadata `group`. Audit đối chiếu cho thấy3 recordings trong merged ít hơn tổng11 samples và pressure có sai khác export tối đa khoảng5×10⁻⁵ source units ở các recordings aligned; vì vậy không dùng arrays merged làm raw input. Các partial fits trước source revision đều được archive và loại khỏi benchmark. Original SVC, merged JSON, baseline và Task1 không bị sửa.

Khác biệt preprocessing/source precision giữa raw branch và protected Kinematic reference là một giới hạn khi diễn giải so sánh; không gán chênh lệch accuracy riêng cho temporal ordering. Trivial majority reference chỉ chọn lớp phổ biến trong outer Train; score0/1 là hard decision, không phải calibrated probability.

'''
    txt=txt.replace('## 3. Preprocessing và alignment',source_note+'## 3. Preprocessing và alignment')
    (OUT/'REPORT_VI.md').write_text(txt,encoding='utf-8');print('VI RAW TEMPORAL REPORT WRITTEN',flush=True)

if __name__=='__main__':main()
