# Phase A — Inventory, provenance và tái tạo cohort

**Verified fact:** audit đọc lại toàn bộ merged files và SVC theo đường dẫn thật trong code. Các con số là ID/files theo từng nguồn, không tự coi các ID khác nhau là những người vật lý khác nhau.

| entity | count |
| --- | --- |
| merged_participants | 294 |
| selected_participants | 258 |
| excluded_merged_participants | 36 |
| merged_recordings | 4000 |
| original_svc_files | 4000 |
| participants_with_svc | 288 |
| raw_metadata_rows | 289 |
| raw_metadata_unique_ids | 288 |
| raw_metadata_duplicate_id_rows | 1 |
| selected_tasks | 14 |
| selected_recordings | 3612 |

| stage | entering | retained | excluded_at_step |
| --- | --- | --- | --- |
| all_merged | 294 | 294 | 0 |
| complete_14_tasks | 294 | 266 | 28 |
| no_duplicate_tasks | 266 | 266 | 0 |
| valid_signal_arrays | 266 | 266 | 0 |
| binary_group | 266 | 258 | 8 |

Điều kiện `prepare.py`: đủ14 canonical tasks, không trùng task, arrays cùng chiều và ít nhất4 samples, không nonfinite, `group`∈{0,1}. `flag_to_omit_total` không tham gia eligibility. Tái tạo IDs/labels khớp `kinematic.npz` và six-fold assignments. Không tạo feature matrix mới, không đổi cohort.

Phát hiện source-link: 12 merged IDs có liên kết nguồn không đầy đủ; 6 trong số đó thiếu cognitive metadata. Không trường hợp nào thuộc cohort258. Có IDs trong CSV nhưng không có SVC và IDs có SVC nhưng không match CSV chính xác; không tự sửa mã hay gán nhãn. `identity_link_anomalies.csv` lưu phép kiểm tra chuỗi và `participant_source_links.csv` lưu liên kết thật. Không có bằng chứng đủ để gộp các ID này thành cùng người.

CSV cognitive có 1 ID lặp. Các khác biệt nằm ở các trường đánh giá HLS trong `duplicate_metadata_ids.csv`; không phải hai dòng hoàn toàn giống nhau. Group, cờ dysgraphia, city/site trên cohort được đối chiếu với mọi giá trị của các dòng lặp, không chọn dòng theo kết quả mô hình.

Kinematic đọc per-sample merged JSON qua `diagramo.branches.kinematic.extract` (alias `features.kinematic`), 74 features/task. Temporal hiện tại đọc **SVC gốc**, 21 channels, NPZfloat32 và packed float16; đây không phải cùng đường dữ liệu với Kinematic. Source migration và sự khác biệt export được giữ nguyên, kiểm toán lại ở Phase D. Normalization chỉ lấy statistics của Train fold.

`metadata_dictionary.csv` thống kê tất cả trường từ CSV và merged: nguồn, dtype, missingness, cardinality, domain. Các trường city, school/site, site_type, kỳ thu thập, age, sex/class/language thật sự tồn tại. Không có bằng chứng local về model thiết bị/serial, clock calibration, firmware hoặc recruitment SOP. Không suy phần cứng từ city/site.

Hash provenance của nguồn đọc nằm trong `source_provenance.json`; `PROTECTED_BEFORE.csv` bao phủ input/cache/checkpoint/code và artifacts được bảo vệ. Không copy raw signals hoặc cognitive item rows vào audit output.
