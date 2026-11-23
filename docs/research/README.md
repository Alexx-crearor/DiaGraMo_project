# Bản đồ nghiên cứu DiaGraMo

Đọc theo thứ tự: mốc XAI trước → Temporal/locked OOF → validity/provenance → independent opaque benchmark → quyết định tuần tới.

- [Tiến trình theo Git](RESEARCH_TIMELINE.md).
- [Ba kiến trúc và phạm vi khác nhau](ARCHITECTURE_EVOLUTION.md).
- [Registry có nguồn metric và commit](EXPERIMENT_REGISTRY.csv).
- [Mục lục report và bảng gốc](RESULTS_INDEX.md).
- [Giới hạn khoa học](VALIDITY_AND_LIMITATIONS.md).
- [Báo cáo tuần](../weekly_reports/2026-11-16_to_2026-11-22/WEEKLY_PROGRESS_VI.md).

## Điểm vào theo giai đoạn

- Original reproduction: [results/REPORT_VI.md](../../results/REPORT_VI.md) — 168 historical CV rows; Kinematic/MOMENT/GAF/CWT.
- Previous XAI milestone: [results/kinematic_xai/REPORT_VI.md](../../results/kinematic_xai/REPORT_VI.md) — Model explanation; not causal biomarkers.
- Initial raw temporal: [results/raw_temporal/REPORT_VI.md](../../results/raw_temporal/REPORT_VI.md) — Historical exploratory architectures.
- Recovery development: [results/raw_temporal_v2/REVIEW_SNAPSHOT_VI.md](../../results/raw_temporal_v2/REVIEW_SNAPSHOT_VI.md) — Development/pilot ledger; not whole-cohort OOF.
- SSL recovery: [results/raw_temporal_v2/PHASE3_RECOVERY_VI.md](../../results/raw_temporal_v2/PHASE3_RECOVERY_VI.md) — INCOMPLETE; smoke/pilot only.
- Lightweight validation: [results/raw_temporal_v2/PHASE4_LITE_REPORT_VI.md](../../results/raw_temporal_v2/PHASE4_LITE_REPORT_VI.md) — Exploratory; distinct from original full nested queue.
- Locked six-fold: [results/raw_temporal_v2/locked_study/LOCKED_RESEARCH_REPORT_VI.md](../../results/raw_temporal_v2/locked_study/LOCKED_RESEARCH_REPORT_VI.md) — Complete matched258 OOF; frozen configuration, exploratory cohort.
- Data validity: [results/data_audit_v3/COMPREHENSIVE_DATA_AUDIT_VI.md](../../results/data_audit_v3/COMPREHENSIVE_DATA_AUDIT_VI.md) — Source–label association; not causal device attribution.
- Provenance: [results/provenance_verification/DATASET_PROVENANCE_VERIFICATION_VI.md](../../results/provenance_verification/DATASET_PROVENANCE_VERIFICATION_VI.md) — Published grouping clarified; individual clinical adjudication incomplete.
- External compatibility: [results/external_validation/EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md](../../results/external_validation/EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md) — Raw exists; exact frozen transfer invalid under current contract.
- Native-channel benchmark: [results/native_channel_research/NATIVE_CHANNEL_TEMPORAL_RESEARCH_REPORT_VI.md](../../results/native_channel_research/NATIVE_CHANNEL_TEMPORAL_RESEARCH_REPORT_VI.md) — Complete five-fold120; exploratory internal CV, NOT frozen transfer.

`src/diagramo/branches/` và pipeline baseline vẫn KEEP_ACTIVE. Code Temporal, audit, provenance, external/native cũng được giữ tại đường dẫn cũ để tái lập; chỉ chạy khi có protocol mới được chủ động lựa chọn. Các report/metric/checkpoint hoàn tất là KEEP_HISTORICAL, không được ghi đè. REVIEW_FOR_REMOVAL là đề xuất cần chủ sở hữu duyệt, không phải danh sách đã xóa.
