# DiaGraMo — tái lập và nghiên cứu biểu diễn chữ viết

Project giữ bản tái lập bốn nhánh Kinematic, MOMENT, GAF, CWT cùng các nghiên cứu XAI, Raw Temporal, validity/provenance và benchmark Slovak. Các nghiên cứu dùng cohort/target khác nhau; kết quả hiện tại phục vụ nghiên cứu thăm dò, chưa xác nhận chẩn đoán lâm sàng hoặc tổng quát hóa độc lập nguồn.

## Đọc nhanh

- [Mốc XAI đã báo cáo](results/kinematic_xai/REPORT_VI.md).
- [Tiến độ tuần 16–22/11/2026](docs/weekly_reports/2026-11-16_to_2026-11-22/WEEKLY_PROGRESS_VI.md) và [nội dung 11 slides](docs/weekly_reports/2026-11-16_to_2026-11-22/PRESENTATION_CONTENT_VI.md).
- [Baseline Kinematic và 168 kết quả tái lập](results/cv/summary.csv).
- [Kiến trúc Temporal thực tế](docs/research/ARCHITECTURE_EVOLUTION.md) và [locked six-fold study](results/raw_temporal_v2/locked_study/LOCKED_RESEARCH_REPORT_VI.md).
- [Data Audit V3](results/data_audit_v3/COMPREHENSIVE_DATA_AUDIT_VI.md) và [provenance/clinical target](results/provenance_verification/DATASET_PROVENANCE_VERIFICATION_VI.md).
- [Khảo sát compatibility Slovak](results/external_validation/EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md) và [opaque benchmark B0–B4](results/native_channel_research/NATIVE_CHANNEL_TEMPORAL_RESEARCH_REPORT_VI.md).
- [Quyết định và kế hoạch tiếp theo](docs/weekly_reports/2026-11-16_to_2026-11-22/NEXT_WEEK_PLAN_VI.md).

## Tái lập gốc và bốn nhánh

Bản dựng độc lập từ mô tả bài *Towards Objective Dysgraphia Detection: A Multi-Branch Deep Learning Approach for Online Handwriting Analysis*. Chi tiết công thức, encoder và CV: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), [báo cáo baseline](results/REPORT_VI.md), [đối chiếu bảng bài báo](results/PAPER_TABLES_VI.md).

| Nhánh | Code giữ nguyên | Biểu diễn |
|---|---|---|
| Kinematic | `src/diagramo/branches/kinematic.py` | 74 features/task, 1.036/người |
| MOMENT | `src/diagramo/branches/moment.py` | Frozen MOMENT-1-large, 14.336/người |
| GAF | `src/diagramo/branches/gaf.py` | GASF 7×224×224, frozen image encoder |
| CWT | `src/diagramo/branches/cwt.py` | Morlet 224 scales, frozen image encoder |

Encoder ảnh trong `scripts/extract.py`; công thức/API ở `src/diagramo/features.py`; weights/hash trong [manifest](references/weights_manifest.json). Original Early Fusion vẫn là nối vector; không tái đưa experimental fusion architecture đã bỏ vào baseline.

## Dataset và target

DiaGraMo giữ 258 participant đủ14canonical tasks, target `group` với115/143. `group` là broader handwriting-disability grouping, không đồng nhất dysgraphia-specific flag hoặc confirmed diagnosis từng người. Năm disagreement và ba omission flags được giữ theo protocol, không sửa sau khi xem kết quả.

Stratified6-Fold theo participant, seed42; mọi fit preprocessing chỉ trong Train. City/source có association mạnh với nhãn; participant CV không tương đương source-independent validation. [Giới hạn và cách diễn đạt](docs/research/VALIDITY_AND_LIMITATIONS.md).

## Kết quả nghiên cứu hiện tại

| Cohort và protocol | Model | Macro-F1 OOF |
|---|---|---:|
| DiaGraMo258, sáu folds | Kinematic LightGBM | 0,847163 |
| DiaGraMo258, sáu folds | Stroke Temporal | 0,713178 |
| DiaGraMo258, sáu folds | Fixed50 probability fusion | 0,840681 |
| Slovak120, năm folds | Statistical LightGBM B2 | 0,932413 |
| Slovak120, năm folds | Ordered CNN B3 | 0,941337 |
| Slovak120, năm folds | Permuted CNN B4 | 0,941337 |

Không xếp hạng chéo hai cohort. Fixed50−Kinematic F1 CI chứa0; B3/B4 cùng binary predictions; B3−B2 CI cũng chứa0. Slovak là internal exploratory benchmark, **không phải frozen external transfer của DiaGraMo**. Nguồn số, protocol, trạng thái và commit có trong [EXPERIMENT_REGISTRY.csv](docs/research/EXPERIMENT_REGISTRY.csv).

## XAI, Temporal và independent research

XAI là mốc trước tuần này: TreeSHAP, task/group attribution, permutation, LOTO và Train-only API; notebook05 giữ nguyên. Attribution của `copy` giải thích fitted model, không chứng minh biomarker hoặc số task tối thiểu.

Raw Temporal dùng7raw→21processed channels, windows≤128, multi-scale CNN, relative segment attention, task identity và learned participant pooling, khoảng25.252 tham số. Không gọi task pooling là task-level Transformer. [Timeline](docs/research/RESEARCH_TIMELINE.md) tách development, Lite, locked six folds và SSL pilot-only/incomplete.

Slovak120 có raw c0–c6 nhưng thiếu verified exporter units/taskcutpoints. CNN13.761 tham số dùng windows256 và order-permutation control, khác Stroke Temporal. Cả hai cohort còn giới hạn recruitment/source confounding.

## Cấu trúc và hướng đọc

```text
src/diagramo/branches/       bốn nhánh gốc
src/diagramo/temporal*/      raw temporal riêng
scripts/                    CLI cũ giữ đường dẫn
scripts/{data_audit_v3,provenance_verification,external_validation,native_channel_research}/
scripts/research_cleanup/   inventory, registry, docs, preservation và checks
configs/                    cấu hình nghiên cứu giữ nguyên
notebooks/                  chín notebook01–09, không ghi lại output
results/                    toàn bộ artifact lịch sử, không di chuyển
references/                 weights provenance
 docs/research/             research index, timeline, registry, limitations
 docs/weekly_reports/       báo cáo tuần, slide content, claim/figure index
 docs/research_cleanup/     inventory, duplicate review, test và hash validation
 data/ và .cache/           nguồn/cache/weights local; Git-ignore
```

[Mục lục nghiên cứu](docs/research/README.md), [kết quả theo giai đoạn](docs/research/RESULTS_INDEX.md), [hình/bảng trình bày](docs/weekly_reports/2026-11-16_to_2026-11-22/FIGURE_INDEX.md). KEEP_ACTIVE là code/entrypoint vẫn dùng; KEEP_HISTORICAL giữ mọi report/checkpoint/OOF. REVIEW_FOR_REMOVAL chỉ là [đề xuất](docs/research_cleanup/REMOVAL_REVIEW_PLAN.md), chưa xóa file nào.

## Môi trường và tái lập

Môi trường đã chạy: Python3.10, PyTorch2.7.1/torchvision0.22.1 CUDA12.8; dependency baseline trong [requirements-lock.txt](requirements-lock.txt), XAI additions trong [requirements-kinematic-xai.txt](requirements-kinematic-xai.txt). Không cài thêm package trong cleanup. GPU local Quadro T1000 4GB; không cần GPU để đọc báo cáo.

Lệnh và entrypoint gốc được giữ trong [legacy reproduction guide](docs/research/LEGACY_ENTRYPOINT_GUIDE.md). Các runner có thể ghi lại artifact khi chủ động tái lập; đọc protocol và tách output trước khi chạy. Không gọi full queue để kiểm tra tài liệu.

Kiểm tra mà không chạy một research experiment mới:

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/research_cleanup/validate.py
.venv/Scripts/python.exe -X utf8 -B scripts/research_cleanup/checks.py --stage after
.venv/Scripts/python.exe -X utf8 -B scripts/research_cleanup/audit.py verify
```

Historical native full-tree hash guard được chạy trước cleanup; sau các source/README edits được cho phép, đối chiếu bằng manifest cleanup mới và giữ nguyên historical manifest. Không sửa report/metric cũ để làm guard xanh. [Bàn giao và test status](docs/research_cleanup/FINAL_CLEANUP_REPORT_VI.md).

## Trạng thái và bước tiếp theo

Full locked six-fold và opaque B0–B4 đã hoàn tất; full SSL/original expensive nested protocol chưa hoàn tất. Ưu tiên source-aware replication và kiểm chứng shortcut/order hypotheses trước khi tăng model capacity. Raw, clinical rows, individual OOF và checkpoint local không được thêm vào Git trong cleanup. Không có retraining hoặc metric thí nghiệm mới.
