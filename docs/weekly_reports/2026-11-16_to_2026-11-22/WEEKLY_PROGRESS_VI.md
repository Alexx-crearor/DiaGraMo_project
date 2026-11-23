# Báo cáo tiến độ nghiên cứu — 16–22/11/2026

## 1. Công việc đã hoàn thành trước mốc báo cáo này

Kinematic Task-Level XAI (`80f9493`) là điểm xuất phát đã báo cáo: TreeSHAP, task/group importance, grouped permutation, LOTO, fold stability và Train-only importance. Baseline Kinematic Macro-F1≈0,8472; `copy` đứng đầu SHAP task mass ở sáu folds. Đây là hành vi mô hình, không causal clinical biomarker và không đủ để kết luận số task tối thiểu. XAI gợi câu hỏi: thống kê74features/task có bỏ lỡ diễn biến trong mẫu viết không?

Nguồn: [XAI report](../../../results/kinematic_xai/REPORT_VI.md) và [task importance](../../../results/kinematic_xai/exploratory/tables/task_importance.csv).

## 2. Mục tiêu nghiên cứu trong tuần

Kiểm tra raw chronological representations; phát triển Stroke-level relative attention và thử fixed fusion với Kinematic. Sau khi kết quả không vượt baseline, tập trung kiểm toán target/nguồn thay vì tăng capacity. Tìm cohort độc lập để kiểm tra giới hạn phương pháp; không mặc định external data tương thích frozen model.

## 3. Kiến trúc Temporal thực tế

Bảy tín hiệu SVC →21 processed channels →non-overlap windows≤128 →multi-scale CNN kernels5/13/29 →bốn local tokens/window →segment aggregation →RelativeStrokeBlock hai heads →task identity embeddings14 →learned participant pooling →pT. Khoảng25.252 tham số. Relative attention ở segment/stroke level; task pooling không phải task-level Transformer. Kinematic xuất pK; fusion cố định pF=0,5pK+0,5pT.

Nguồn code: [MultiScaleLocal/RelativeStrokeBlock/MultiScaleModel](../../../src/diagramo/temporal_v2/multiscale.py); [sơ đồ và phân biệt ba tuyến](../../research/ARCHITECTURE_EVOLUTION.md).

## 4. Kết quả chính trên DiaGraMo

| Model | Macro-F1 OOF | Accuracy | AUC |
| --- | --- | --- | --- |
| Kinematic | 0.847163 | 0.848837 | 0.906902 |
| Temporal | 0.713178 | 0.713178 | 0.807540 |
| Fixed50 | 0.840681 | 0.841085 | 0.906111 |


Đủ258 người/sáu folds, participant stratification seed42; cùng người/folds khi so sánh. Fixed50−Kinematic ΔMacro-F1=-0.006483, paired CI95%[-0.053470,0.041286], gồm0. Fusion tăng ở3/6folds, giảm3/6; corrected20/harmed22. Temporal thấp hơn Kinematic; fixed fusion không chứng minh cải thiện. Không diễn giải SD như CI. Fold mean±SD ở bảng gốc; paired bootstrap2000 lần trong lớp, giữ fitted models cố định, không sửa source/researcher-selection bias.

Nguồn: [OOF metrics](../../../results/raw_temporal_v2/locked_study/matched_oof_metrics.csv), [fold mean±SD](../../../results/raw_temporal_v2/locked_study/fold_mean_sd.csv), [paired CI](../../../results/raw_temporal_v2/locked_study/paired_bootstrap.csv).

## 5. Data validity và source confounding

| Nguồn đã gộp | group0 | group1 | Tổng |
|---|---:|---:|---:|
| City1 |115|7|122|
| Các city khác |0|136|136|
| Tổng |115|143|258|

15city categories; Cramér’s V≈0,946827. City-only Train-mapping control đạt OOF Accuracy≈0,972868, cao hơn class prior≈0,554264. Đây là khả năng dự đoán dataset target từ nguồn, không clinical accuracy. Participant-independent CV vẫn trộn nguồn giữa Train/Test. Chưa chứng minh direct ID leakage hoặc causal device effect; thiếu overlap hai lớp tại nhiều nguồn làm việc tách motor/source effects chưa khả thi.

Nguồn: [contingency](../../../results/data_audit_v3/source_label_contingency.csv), [negative controls](../../../results/data_audit_v3/metadata_shortcut_metrics.csv), [association statistics](../../../results/data_audit_v3/source_association_metrics.csv).

## 6. Dataset provenance và clinical label

Đã nối data descriptor và Zenodo release trong Phase5. `group` là handwriting-disability grouping rộng; dysgraphia-specific flag và ongoing clinical/educational assessment không phải cùng biến. Năm disagreement quan sát được, một phần tương thích mô tả đang assessment, nhưng individual clinical adjudication, một số rules CSV/RDS→merged và duplicate handling chưa có xác minh độc lập. Không đổi nhãn hay loại omission cases sau khi xem kết quả.

Nguồn: [provenance report](../../../results/provenance_verification/DATASET_PROVENANCE_VERIFICATION_VI.md).

## 7. Khảo sát cohort Slovak độc lập

Drotár & Dobeš2020:120 người,57DYSGR/63controls;121SVC,120recording hợp lệ,2.176.419raw samples. Một file rỗng nhưng người tương ứng còn recording hợp lệ. Bảy numeric columns không header, units/exporter order chưa VERIFIED; tám điều kiện trong paper không có raw task cutpoints và không khớp14canonical tasks. Frozen DiaGraMo transfer hiện không hợp lệ theo contract. Recruitment case/control cũng đi qua institutional pathways khác nhau; không có site metadata cá nhân để source-held-out validation.

Nguồn: [external compatibility report](../../../results/external_validation/EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md).

## 8. Native-channel benchmark B0–B4

| Model | Macro-F1 OOF | Accuracy | AUC | Balanced Accuracy |
| --- | --- | --- | --- | --- |
| B0 | 0.344262 | 0.525000 | 0.479950 | 0.500000 |
| B1 | 0.598214 | 0.600000 | 0.592175 | 0.604845 |
| B2 | 0.932413 | 0.933333 | 0.960457 | 0.929825 |
| B3 | 0.941337 | 0.941667 | 0.969368 | 0.940267 |
| B4 | 0.941337 | 0.941667 | 0.971039 | 0.940267 |


B0 reuse prior; B1 sample-length LightGBM; B2 opaque statistical LightGBM; B3 ordered compact CNN; B4 cùng CNN với fixed synchronized valid-row permutation. Giữ đúng120 người/five participant folds Phase6, base seed42, no outer-Test selection; 20epochs CNN cố định. B3 và B4 có cùng120binary predictions: ΔF1=0.000000, corrected=0, harmed=0. ΔAUC=-0.001671, CI95%[-0.006118,0.002228]. CI F1=[0,0] suy biến do fixed hard-prediction equality, không population equivalence. B3−B2 ΔF1=0.008923, CI95%[-0.025584,0.044403], chưa chứng minh CNN vượt statistical baseline.

Macro-F1≈94,13% không phải successful DiaGraMo external transfer. Một main seed; physical schema chưa rõ; B2 có28difference summaries phụ thuộc thứ tự; B4 có thể giữ implicit-position cues. Chưa có evidence disease-specific temporal mechanism.

Nguồn: [metrics](../../../results/native_channel_research/MODEL_METRICS.csv), [paired contrasts](../../../results/native_channel_research/PAIRED_MODEL_COMPARISON.csv), [report](../../../results/native_channel_research/NATIVE_CHANNEL_TEMPORAL_RESEARCH_REPORT_VI.md).

## 9. Bài học khoa học

- Tích cực: hoàn thành full matched six-fold evaluation, audit dữ liệu/clinical provenance và benchmark cohort thứ hai có permutation control.
- Âm tính: Temporal không vượt Kinematic, fixed fusion không cải thiện đáng tin; ordered CNN không có ưu thế phân loại so với permuted control.
- Chưa kết luận: CNN hơn statistics, motor-vs-source contribution, physical channel meanings và clinical/source-independent validity.
- SSL chỉ smoke/pilot, trạng thái **INCOMPLETE**; original expensive nested Phase4 không được chạy trong tuần cleanup.
- Bước cleanup này chỉ đọc/kiểm thử/tổ chức lại; không retrain, tối ưu hoặc tạo metric thí nghiệm mới.

## 10. Kế hoạch tuần tới

Ưu tiên replication/source-aware data design và khóa shortcut/implicit-position hypothesis trước score. Development-only ablation marginal-only, channel/c2 removal và order robustness là lựa chọn nhẹ sau khi protocol được duyệt; chưa chạy trong task này. Chưa ưu tiên Transformer lớn hơn. Chi tiết ở [NEXT_WEEK_PLAN_VI.md](NEXT_WEEK_PLAN_VI.md).
