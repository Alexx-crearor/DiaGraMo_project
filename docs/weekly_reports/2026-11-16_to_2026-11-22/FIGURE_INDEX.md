# Figure/table index — giữ nguyên ảnh cũ

Đã xem trực quan task_importance.png, p2_architecture.png và source_label_composition.png; đối chiếu kiến trúc với code. Không render lại ảnh lịch sử. Các bảng native/locked được đọc từ CSV gốc. READY là đủ dùng nội dung, không phải chứng nhận quyền phát hành lâm sàng.

| Figure/table | Source | Stage | Message | Status | Privacy | Slide |
|---|---|---|---|---|---|---|
| XAI task mass | [results/kinematic_xai/figures/task_importance.png](../../../results/kinematic_xai/figures/task_importance.png) | Previous XAI | copy dominates fitted model attribution | READY | No direct ID; aggregate across258 | 1 |
| XAI→Temporal transition | [docs/research/ARCHITECTURE_EVOLUTION.md](../../research/ARCHITECTURE_EVOLUTION.md) | Cross-stage | Three distinct pipelines | READY (Mermaid; slide redraw optional) | No individual data | 2 |
| Stroke architecture | [results/raw_temporal_v2/figures/p2_architecture.png](../../../results/raw_temporal_v2/figures/p2_architecture.png) | Recovery/locked | M3 relative stroke attention; task pooling | READY; highlight M3 | No individual data | 3 |
| Matched Kinematic/Temporal/Fixed50 | [results/raw_temporal_v2/locked_study/matched_oof_metrics.csv](../../../results/raw_temporal_v2/locked_study/matched_oof_metrics.csv) | Locked258 | Temporal/fusion do not beat baseline | READY table | Aggregate only | 5 |
| Fusion paired uncertainty | [results/raw_temporal_v2/locked_study/paired_bootstrap.csv](../../../results/raw_temporal_v2/locked_study/paired_bootstrap.csv) | Locked258 | Delta CI contains zero | READY table | Aggregate only | 5 |
| Source-label composition | [results/data_audit_v3/figures/source_label_composition.png](../../../results/data_audit_v3/figures/source_label_composition.png) | AuditV3 | Strong source association, not causality | READY internal; prefer City1/other grouping for wider release | No IDs; rare city counts need disclosure review | 6 |
| Dataset compatibility | [results/external_validation/CHANNEL_COMPATIBILITY_MATRIX.csv](../../../results/external_validation/CHANNEL_COMPATIBILITY_MATRIX.csv) | External | Raw availability does not imply frozen contract match | READY table | Aggregate only | 8 |
| Slovak B0–B4 | [results/native_channel_research/MODEL_METRICS.csv](../../../results/native_channel_research/MODEL_METRICS.csv) | Native120 | High internal scores; distinct cohort | READY table | Aggregate only | 9 |
| B3−B4 paired control | [results/native_channel_research/PAIRED_MODEL_COMPARISON.csv](../../../results/native_channel_research/PAIRED_MODEL_COMPARISON.csv) | Native120 | Same hard predictions; AUC difference uncertain | READY table | Aggregate only | 10 |
| Research roadmap | [docs/weekly_reports/2026-11-16_to_2026-11-22/NEXT_WEEK_PLAN_VI.md](NEXT_WEEK_PLAN_VI.md) | Planning | Source-aware replication before capacity | READY table | No individual data | 11 |
| Participant local SHAP/waterfalls | [results/kinematic_xai/figures/waterfall_false_negative.png](../../../results/kinematic_xai/figures/waterfall_false_negative.png) | Previous XAI | Individual explanation not required for weekly narrative | NOT RECOMMENDED; privacy review required | Individual attribution/clinical prediction even if ID absent | Backup only |

Không đưa participant_sensor_signatures, embeddings scatter có individual points, raw writing traces, local SHAP cases hoặc participant_complementarity rows vào deck mặc định. B0–B4 và matched results dùng bảng có sẵn nên không cần tạo PNG mới.
