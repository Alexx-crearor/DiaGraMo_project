# Nguồn gốc dataset và tài liệu gốc
Nguồn chính: [data descriptor](https://doi.org/10.1038/s41597-026-08625-x), [Zenodo v2](https://zenodo.org/records/21236910). Kiểm tra 2026-11-21T16:59:34.309386+00:00. Attribution: Zvončáková, Mekyska, Klocek, Mucha & Galáž (2026); dataset CC BY 4.0. Bảng kết quả local trỏ về Data Audit V3 tại commit f419530; không tái chạy V3.


**VERIFIED — nguồn công bố.** Concept DOI `10.5281/zenodo.18299326` hiện trả record v2 `21236910`; API xác nhận CC BY 4.0. V2 có ZIP 1.357.028.237 bytes với MD5 công bố `e9d7a8bc5c98d1edf88824fbc90b06ca`. Không có ZIP local được đối chiếu checksum, nên không tuyên bố local là bản sao byte-identical của v2. Không tải archive hay dữ liệu cá nhân mới.

**VERIFIED — tài liệu có thể truy cập.** Đã đọc PDF data descriptor 12 trang, Zenodo HTML/API, README hai công cụ BDALab và code cohort local. `info/README.txt`, `Codebook.xlsx`, `metadata.json`, `distorted_data.xlsx` và bốn R scripts được bài báo chỉ rõ nhưng không có trong project hiện tại. Preview archive báo quá nhiều files; container endpoint403. Vì archive chứa dữ liệu trẻ em, không tải toàn bộ chỉ để lấy tài liệu. Cần chủ dữ liệu cung cấp riêng codebook/R scripts không chứa hồ sơ cá nhân. Ghi blocker, không xem mô tả bài báo là đã kiểm tra source code.

| item | documented_evidence | source | verification_status | remaining_gap |
| --- | --- | --- | --- | --- |
| Population/recruitment | Schools and pedagogical-psychological counselling centres; different routes for HD and TD | S01 pp.2–3 | VERIFIED | No city-by-city sampling frame or referral denominator |
| City identity | Pseudonymized city labels in released metadata | S01 Ethics; S13 | VERIFIED | Real cities and device mappings unavailable; city is not hospital/device |
| Clinical source | Prior PPCC records and school anamnestic reports; includes ongoing diagnoses | S01 Assignment of diagnosis | VERIFIED | Participant-level diagnostic records not released |
| Device family | Wacom Cintiq 16 DTK-1660 / Pro Pen 2 KP504E; HandAQUS | S01 p.5; S04 | VERIFIED | Per-session serial/firmware/driver histories absent |
| Calibration | Factory digitizer calibration and pen-display alignment described | S01 p.5 | SUPPORTED | No calibration certificate, pressure equivalence or session logs inspected |
| Operator | Individual administration by study personnel described; named visual QC evaluators | S01 Methods | SUPPORTED | No operator-to-participant/site mapping or inter-rater linkage |
| Task protocol | 16filename variants; local14canonical tasks collapse grade-specific dictation/copy | S01 Table3; S12 TASKS/canonical | VERIFIED | Same canonical task does not imply identical grade-specific stimuli |
| Collection period | December2021–June2023; dates released as quarter/year | S01 Participants; S02 Notes | VERIFIED | Site-specific batch/protocol drift not resolved |
| Timestamp/spatial units | Paper documents seconds and millimetres before JSON conversion | S01 Data Preprocessing p.5 | SUPPORTED | Not independently certified clock/spatial calibration for local sessions |
| Sampling cadence | Paper average 167 Hz; V3 local median-derived cadence~200/source-time-unit | S01 p.5; S13 PhaseD | CONTRADICTORY | Mean-vs-median estimators differ; reconcile distributions/clock provenance before causal inference |

**SUPPORTED — đối chiếu phiên bản.** Local không có trường ngày sinh/ngày ký consent theo tên cột trong hai CSV; điều này phù hợp privacy notes v2, nhưng không đủ chứng minh phiên bản hoặc kiểm tra mọi JSON/RDS. 294 released metadata IDs, 276 mô tả trong bài và 258 cohort nghiên cứu là ba tập có tiêu chí khác nhau. Không lấy số người khớp gần nhau để nhận diện cohort.

**UNVERIFIED — tài liệu bổ sung.** Không truy được codebook cấp biến, nhật ký thiết bị, bản anamnestic hoặc rule lựa chọn duplicate. Không suy ra tên thành phố từ prefix ID. Những paper downstream dùng DiaGraMo không thay thế chứng cứ tuyển mẫu của tác giả dữ liệu.
