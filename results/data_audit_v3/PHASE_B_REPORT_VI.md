# Phase B — Label validity

`group` là target thực tế. `diagnosis` trong audit là **alias của `special_care_type_dysgraphia`**, không phải một nhãn lâm sàng độc lập mới phát hiện. Dữ liệu nhị phân và quan hệ giữa chúng được xác minh; định nghĩa chẩn đoán/cách tuyển nhóm chưa có codebook độc lập để xác nhận.

| field | origin | implementation | clinical_definition |
| --- | --- | --- | --- |
| group | CSV and merged cognitive metadata | Actual target for protected models | UNVERIFIED: grouping/case ascertainment and screening rules absent locally |
| diagnosis | Audit alias of special_care_type_dysgraphia | Not an independently verified diagnosis variable | No separate exact diagnosis field assumed; alias explicitly distinguished |
| special_care_type_dysgraphia | CSV and merged metadata | Specific binary flag, distinct from group | UNVERIFIED: clinician, diagnostic tool, threshold and assessment timing unknown |
| flag_to_omit_total | Merged metadata; absent from original cognitive CSV | Recorded but not used by prepare.py eligibility | Flag components observable in merged; transformation/author intent/official exclusion rule UNVERIFIED |

| group | diagnosis0 | diagnosis1 |
| --- | --- | --- |
| 0.000000 | 114 | 1 |
| 1.000000 | 4 | 139 |

Có 5 participant hai trường không khớp và 3 participant mang omission flag. Không thể tự suy trường nào đúng lâm sàng. `label_disagreements.csv` chứa riêng các disagreement; `omission_flag_details.csv` chứa flagged participants cùng các component flags quan sát; `label_consistency.csv` là bảng đầy đủ. Kiểm tra OR của component flags với total chỉ là tính nhất quán quan sát, không thay cho codebook/quy tắc chính thức; không loại người hoặc sửa nguồn.

**Phát hiện provenance mới:** raw cognitive CSV không có `flag_to_omit_total` và các omission flags như merged. Vì vậy không được ghi “đã đối chiếu flag với CSV gốc”. `metadata_join_checks.csv` đánh dấu trường vắng mặt là UNVERIFIABLE, không cho pass giả. Processed RDS tồn tại nhưng local không có pyreadr/Rscript để giải mã; không cài/tải qua mạng. Cần xác nhận pipeline tạo flags và ý nghĩa exclusion từ chủ dữ liệu. Các component flags hiện có là quan sát, không tự coi tổng OR là quy tắc chính thức nếu thiếu code tạo cờ.

Sensitivity dưới đây chỉ dùng cùng xác suất OOF đã huấn luyện cho `group`. Các cohort253/255 và nhãn diagnosis là phân tích hậu nghiệm, không phải retraining hoặc independent validation, không dùng chọn winner.

| model | cohort | target | n | group0 | group1 | accuracy | f1_macro | auc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kinematic_LightGBM | original_258 | label | 258 | 115 | 143 | 0.848837 | 0.847163 | 0.906902 | 0.122450 |
| Kinematic_LightGBM | consistent_labels | label | 253 | 114 | 139 | 0.849802 | 0.848321 | 0.908620 | 0.121335 |
| Kinematic_LightGBM | no_omission_flag | label | 255 | 113 | 142 | 0.850980 | 0.849291 | 0.908949 | 0.120458 |
| Kinematic_LightGBM | diagnosis_relabelled_same_OOF | diagnosis | 258 | 118 | 140 | 0.837209 | 0.835788 | 0.892797 | 0.132608 |
| Stroke_Temporal | original_258 | label | 258 | 115 | 143 | 0.713178 | 0.713178 | 0.807540 | 0.224193 |
| Stroke_Temporal | consistent_labels | label | 253 | 114 | 139 | 0.719368 | 0.719368 | 0.811814 | 0.218013 |
| Stroke_Temporal | no_omission_flag | label | 255 | 113 | 142 | 0.713725 | 0.713708 | 0.807491 | 0.223400 |
| Stroke_Temporal | diagnosis_relabelled_same_OOF | diagnosis | 258 | 118 | 140 | 0.717054 | 0.717016 | 0.806416 | 0.220331 |
| Fixed50 | original_258 | label | 258 | 115 | 143 | 0.841085 | 0.840681 | 0.906111 | 0.129793 |
| Fixed50 | consistent_labels | label | 253 | 114 | 139 | 0.845850 | 0.845502 | 0.908747 | 0.126895 |
| Fixed50 | no_omission_flag | label | 255 | 113 | 142 | 0.839216 | 0.838730 | 0.905896 | 0.129532 |
| Fixed50 | diagnosis_relabelled_same_OOF | diagnosis | 258 | 118 | 140 | 0.837209 | 0.836964 | 0.897579 | 0.132940 |

Câu hỏi cần chủ dữ liệu: group định nghĩa bằng chẩn đoán hay recruitment/screening nào; dysgraphia flag do ai đánh giá, công cụ/ngưỡng và thời điểm nào; giải thích từng disagreement; nguồn tạo các omission flags và cohort paper chính thức; xử lý duplicate cognitive ID và HLS rows nào được merge.
