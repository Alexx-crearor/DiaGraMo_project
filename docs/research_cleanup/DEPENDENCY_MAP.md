# Dependency và artifact contracts

`IMPORT_EDGES.csv` là toàn bộ import edges tĩnh; `IMPORT_DEPENDENCY_CHECKS.json` ghi syntax, hardcoded-path locations và report-name collisions. AST không chứng minh dynamic imports resolve; CLI/import smoke và tests được ghi riêng. `NOTEBOOK_INVENTORY.json` kiểm tra cả9notebooks nhưng không execute để tránh ghi lại lịch sử.

| Nhóm | Code đọc | Inputs | Outputs lịch sử | Quy tắc |
|---|---|---|---|---|
| Original4branches | prepare/extract/evaluate/run_all; diagramo.branches/features | merged JSON, weights, baseline cache | results/cv, audit, figures | Không đổi công thức/split/encoder hoặc tái thêm experimental fusion |
| Kinematic XAI | kinematic_xai_common, diagramo.kinematic_xai | baseline OOF/checkpoints/schema | results/kinematic_xai | SHAP/raw-vs-probability đúng code, giữ task mapping |
| Temporal | raw_temporal*; temporal/temporal_v2 | original SVC,21channels, Train-only stats | raw_temporal*, locked_study | Segment attention khác task-level Transformer |
| AuditV3 | scripts/data_audit_v3 | metadata/source/cached arrays/OOF | results/data_audit_v3 | Diagnostic, không đổi clinical label |
| Provenance | scripts/provenance_verification | source docs + local metadata | results/provenance_verification | Explicit file-path imports; không import nhầm report.py |
| External | scripts/external_validation | pinned ZIP/XLSX/native reader | results/external_validation | Parser khác DiaGraMo physical preprocessing, không gộp semantics |
| Native | scripts/native_channel_research | Phase6 source/folds; private checkpoints/OOF | results/native_channel_research | Package-relative imports; no original14task head |
| Weekly cleanup | scripts/research_cleanup | read-only authoritative metrics/history | docs/research*, weekly_reports | Không gọi training runners |

`report.py` tồn tại ở nhiều thư mục. Baseline import bằng đường dẫn scripts; audit/provenance/external dùng explicit spec_from_file_location; native dùng relative package import. Giữ namespaces riêng; không deduplicate metrics superficially vì target, threshold, bootstrap và schema khác nhau. Hai requirements files có vai trò baseline lock và XAI additions, không coi là duplicate để xóa.
