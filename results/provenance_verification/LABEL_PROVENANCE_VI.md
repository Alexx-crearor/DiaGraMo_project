# Nhãn: định nghĩa công bố và giá trị local
Nguồn chính: [data descriptor](https://doi.org/10.1038/s41597-026-08625-x), [Zenodo v2](https://zenodo.org/records/21236910). Kiểm tra 2026-11-21T16:59:34.309386+00:00. Attribution: Zvončáková, Mekyska, Klocek, Mucha & Galáž (2026); dataset CC BY 4.0. Bảng kết quả local trỏ về Data Audit V3 tại commit f419530; không tái chạy V3.


**VERIFIED:** Bài gốc đã có mục Assignment of diagnosis; V3 chỉ dùng local nên thiếu bằng chứng này. Nhãn không phải một test chẩn đoán mới do nhóm dataset thực hiện. Không gọi tất cả `group=1` là confirmed Dysgraphia. Cách gọi được đề nghị: **nhóm handwriting disability theo dataset, từ hồ sơ lâm sàng/giáo dục có sẵn, bao gồm trường hợp đang đánh giá**.

**SUPPORTED:** local numeric0 tương ứng nhóm đối chứng,1 tương ứng nhóm HD rộng; codebook ánh xạ chính thức chưa được đọc. Phân biệt `special_care_type_dysgraphia` và alias `diagnosis` của V3. Tên cột không chứng minh một chẩn đoán độc lập.

| field | original_definition | source | evidence | verification_status | limit |
| --- | --- | --- | --- | --- | --- |
| group | HD/TD from prior anamnestic PPCC diagnosis records, including ongoing diagnosis | S01 p.3; S12; cleaned/raw CSV | Published definition verified; local values preserved; five specific-flag discordances | VERIFIED | Numeric 0/1 mapping SUPPORTED by local cross-tabs; author codebook and individual adjudication still absent |
| special_care_type_dysgraphia | Specific special-care indicator in metadata | Local raw/cleaned CSV; merged cases; S01 clinical context | Binary field and relation to free-text care categories observed | SUPPORTED | Not a separately verified clinician diagnosis or standardized threshold |
| diagnosis | V3 audit alias of special_care_type_dysgraphia | S13; scripts/data_audit_v3/phase_abc.py | Alias definition traceable in local code | VERIFIED | Do not describe as independent clinical ground truth |
| flag_to_omit_total | Composite eligibility flag; user-selectable inclusion | S01 pp.3,6,8; cleaned CSV | Field exists in cleaned CSV, absent from raw CSV; prepare.py does not exclude it | SUPPORTED | Exact R expression/default switch not inspected; empirical OR is not a rule proof |
| all_group1_confirmed_dysgraphia | Unsupported claim for every positive protected target | S01 ongoing-diagnosis definition; five-case read-only reconciliation | Broader HD category includes ongoing cases; specific flag differs | CONTRADICTORY | Use dataset-defined HD/group label with clinical-record provenance, not uniformly confirmed dysgraphia |
| label_independence_from_predictors | Not proven at individual historical diagnosis level | S01 pre-existing ascertainment; S12 predictor selection | Current pipeline excludes cognitive/special-care metadata from handwriting predictors | SUPPORTED | No evidence current cohort handwriting scores generated group; historical handwriting assessment may be part of diagnosis |

## Năm bất đồng

Đã join từng trường hợp trên raw CSV, cleaned CSV và merged JSON: True. Không phải sai lệch mới phát sinh giữa ba lớp lưu trữ ở các trường liên quan đã so sánh. Ba trường hợp có cờ ongoing: **SUPPORTED** về tương thích với định nghĩa nhóm rộng. Một trường hợp nhóm đối chứng nhưng cờ dysgraphia, và một trường hợp nhóm case với các cờ rối loạn học tập khác cần tác giả phân xử; không suy đoán động cơ gán nhãn. Giải thích hợp lý không tương đương xác minh hồ sơ cá nhân.

Bảng công khai `DISAGREEMENT_CASE_DECISIONS.csv` cho kết luận riêng từng case reference; không xuất ID, tuổi, site hay raw special-care rows. Mapping và trường liên quan nằm ở `data/provenance_verification_local/UNRESOLVED_LABEL_CASES.csv`, đã Git ignore. Tất cả năm case vẫn UNRESOLVED về adjudication cá nhân. Không sửa nhãn hoặc dùng kết quả này lựa chọn lại mô hình.

**SUPPORTED về tính độc lập:** nguồn nhãn là hồ sơ có trước; pipeline hiện tại không dùng metadata chẩn đoán/cognitive làm predictors. **UNVERIFIED:** từng người được đánh giá lịch sử bằng chính nội dung viết nào, assessors có biết các kết quả đo hay không, mức đồng nhất giữa site. Vì vậy chưa chứng minh hoàn toàn target independence theo nghĩa clinical ascertainment.
