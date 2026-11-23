# Nội dung trình bày — 11 slides, chưa tạo PowerPoint

Mỗi slide giữ riêng implementation, evaluation và inference. XAI là mốc trước; không gộp hai cohort vào một leaderboard.

## Slide 1. Điểm xuất phát: XAI đã báo cáo

- Baseline Kinematic F1≈0,8472.
- TreeSHAP/task/group/permutation/LOTO đã hoàn tất trước mốc này.
- copy nổi bật nhưng attribution không phải causal biomarker.

Hình/bảng: [results/kinematic_xai/figures/task_importance.png](../../../results/kinematic_xai/figures/task_importance.png). Nguồn chính xác: [results/kinematic_xai/exploratory/tables/task_importance.csv](../../../results/kinematic_xai/exploratory/tables/task_importance.csv).

**Thông điệp:** XAI đặt câu hỏi mới, không phải kết quả mới của tuần.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 2. Vì sao thử raw temporal?

- 74features/task nén một chuỗi dài.
- Giả thuyết: chronological structure có thể bổ sung thông tin.
- Không mặc định mô hình lớn sẽ tốt hơn.

Hình/bảng: [docs/research/ARCHITECTURE_EVOLUTION.md](../../research/ARCHITECTURE_EVOLUTION.md). Nguồn chính xác: [src/diagramo/temporal/preprocessing.py](../../../src/diagramo/temporal/preprocessing.py).

**Thông điệp:** Động cơ là kiểm chứng biểu diễn, chưa phải chứng minh temporal biomarker.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 3. Kiến trúc đã triển khai

- 7raw→21processed channels; windows≤128.
- Multi-scale CNN5/13/29, segment-relative attention.
- 14task identities và learned participant pooling; khoảng25.252parameters.
- Task pooling không phải task-level Transformer.

Hình/bảng: [results/raw_temporal_v2/figures/p2_architecture.png](../../../results/raw_temporal_v2/figures/p2_architecture.png). Nguồn chính xác: [src/diagramo/temporal_v2/multiscale.py](../../../src/diagramo/temporal_v2/multiscale.py).

**Thông điệp:** Chỉ rõ nhánh M3 trong hình; tách attention segment khỏi task aggregation.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 4. Giao thức đánh giá và trạng thái

- 258participants;6folds seed42 giữ nguyên.
- Frozen final configuration, Train-only preprocessing.
- Full sixfold OOF hoàn tất; cohort đã inspected nên exploratory.
- SSL chỉ smoke/pilot; full original nested queue chưa hoàn tất.

Hình/bảng: [docs/research/EXPERIMENT_REGISTRY.csv](../../research/EXPERIMENT_REGISTRY.csv). Nguồn chính xác: [results/raw_temporal_v2/locked_study/matched_fold_metrics.csv](../../../results/raw_temporal_v2/locked_study/matched_fold_metrics.csv).

**Thông điệp:** Completed fits khác với completed original research programme.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 5. Kết quả bất ngờ: Fusion không vượt baseline

- Kinematic0,8472;Temporal0,7132;Fixed500,8407 F1.
- ΔFusion−Kinematic≈−0,00648, CI chứa0.
- Fusion corrected20/harmed22, không chỉ nhìn fold tốt.

Hình/bảng: [results/raw_temporal_v2/locked_study/matched_oof_metrics.csv](../../../results/raw_temporal_v2/locked_study/matched_oof_metrics.csv). Nguồn chính xác: [results/raw_temporal_v2/locked_study/paired_bootstrap.csv](../../../results/raw_temporal_v2/locked_study/paired_bootstrap.csv).

**Thông điệp:** Không có reliable fusion improvement.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 6. Source–label confounding

- City1:115group0/7group1; còn lại0/136.
- City-only OOF Accuracy≈97,29%; V≈0,947.
- Không đồng nhất source association với direct leakage hoặc causal device effect.

Hình/bảng: [results/data_audit_v3/figures/source_label_composition.png](../../../results/data_audit_v3/figures/source_label_composition.png). Nguồn chính xác: [results/data_audit_v3/metadata_shortcut_metrics.csv](../../../results/data_audit_v3/metadata_shortcut_metrics.csv).

**Thông điệp:** Participant-independent chưa đủ cho source-independent validity.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 7. Clinical target thực sự là gì?

- group là broader handwriting-disability grouping.
- Specific dysgraphia flag và ongoing assessment cần phân biệt.
- 5disagreements,3omission cases; không tự sửa label/cohort.

Hình/bảng: [docs/weekly_reports/2026-11-16_to_2026-11-22/WEEKLY_PROGRESS_VI.md](WEEKLY_PROGRESS_VI.md). Nguồn chính xác: [results/provenance_verification/LABEL_VERIFICATION_DECISION.json](../../../results/provenance_verification/LABEL_VERIFICATION_DECISION.json).

**Thông điệp:** Không diễn đạt mọi group1 là confirmed dysgraphia.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 8. Cohort độc lập: có raw nhưng không tương thích frozen model

- Slovak120người,57/63labels,120valid recordings.
- 7numeric columns không header, units/taskcutpoints thiếu.
- 8documented conditions khác14canonical tasks.
- Không load frozen DiaGraMo để giả external validation.

Hình/bảng: [results/external_validation/CHANNEL_COMPATIBILITY_MATRIX.csv](../../../results/external_validation/CHANNEL_COMPATIBILITY_MATRIX.csv). Nguồn chính xác: [results/external_validation/TRANSFER_PROTOCOL_DECISION.json](../../../results/external_validation/TRANSFER_PROTOCOL_DECISION.json).

**Thông điệp:** Compatibility audit là kết quả có ích dù transfer chưa hợp lệ.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 9. Opaque benchmark B0–B4

- Prior0,3443;Length0,5982;Statistics0,9324 F1.
- Ordered CNN và Permuted CNN đều0,9413.
- 5participant folds reused; cùng architecture/epochs của B3/B4.
- Không dùng score để đổi threshold/model.

Hình/bảng: [results/native_channel_research/MODEL_METRICS.csv](../../../results/native_channel_research/MODEL_METRICS.csv). Nguồn chính xác: [results/native_channel_research/LOCKED_CONFIG.json](../../../results/native_channel_research/LOCKED_CONFIG.json).

**Thông điệp:** Điểm cao là cohort-internal exploratory result.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 10. Order control và kết luận thận trọng

- B3/B4 cùng binary predictions trên120người.
- ΔF1=0; ΔAUC CI chứa0.
- B3−B2 F1 CI cũng chứa0.
- B2 có difference features; B4 có thể giữ vị trí ngầm.

Hình/bảng: [results/native_channel_research/PAIRED_MODEL_COMPARISON.csv](../../../results/native_channel_research/PAIRED_MODEL_COMPARISON.csv). Nguồn chính xác: [results/native_channel_research/ERROR_OVERLAP.csv](../../../results/native_channel_research/ERROR_OVERLAP.csv).

**Thông điệp:** Chưa chứng minh lợi ích của order hay temporal biomarker.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.

## Slide 11. Bài học và kế hoạch tiếp theo

- Giữ Kinematic làm baseline; không tăng architecture vô điều kiện.
- Ưu tiên source-aware replication và clinical/provenance constraints.
- Khóa marginal-only/channel/order hypotheses trên development data mới.
- Repository có registry, manifest, tests và report navigation; không xóa lịch sử.

Hình/bảng: [docs/weekly_reports/2026-11-16_to_2026-11-22/NEXT_WEEK_PLAN_VI.md](NEXT_WEEK_PLAN_VI.md). Nguồn chính xác: [docs/research/VALIDITY_AND_LIMITATIONS.md](../../research/VALIDITY_AND_LIMITATIONS.md).

**Thông điệp:** CONDITIONAL GO cho nghiên cứu, NO-GO cho clinical/source-invariant claims.

Ghi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.
