# Duplicate metadata investigation

**VERIFIED:** 1 ID bị lặp trong raw CSV; cùng ID vẫn có hai hàng trong cleaned CSV. Xung đột gồm sáu trường HLS. Không xuất ID hoặc điểm số trong báo cáo công khai. Chứng cứ row-level và ordinal matching nằm ở `data/provenance_verification_local/DUPLICATE_METADATA_LOCAL.csv`.

| raw_rows | cleaned_rows | conflicting_fields | merged_matches_raw_occurrences | protected_cohort_member | relevant_label_conflicts |
| --- | --- | --- | --- | --- | --- |
| 2 | 2 | hlsDictatA_ledgibility/hlsDictatB_effort/hlsDictatC_pageLayout/hlsDictatD_letterFormation/hlsDictatE_alterations/hlsDictat_sum | 2 | False | False |

`merged_matches_raw_occurrences` là thứ tự occurrence1/2 trong các hàng cùngID, không phải số dòng file. Nếu chỉ một occurrence khớp đủ sáu trường thì xác minh được nội dung nào đã được nhúng, **không** chứng minh quy tắc first/last/clinical adjudication. Không có script merge gốc hoặc assessment timestamps để xác định đây là đánh giá lặp hay lỗi nhập. **UNVERIFIED** về lựa chọn bản ghi chính thức. Paper nói đã loại duplicate ở quy trình công bố, trong khi hai CSV local vẫn duplicate: **CONTRADICTORY**, có thể liên quan phiên bản hoặc định nghĩa duplicate; không tự chọn lời giải.

Các trường group/specific flag không xung đột giữa hai hàng theo kiểm tra trên. Trạng thái thuộc cohort phải đọc từ `protected_cohort_member` trong bảng, không suy từ việc ID có merged JSON: kiểm tra hiện tại cho thấy duplicate **không thuộc 258 người bảo vệ**. Chưa có bằng chứng trực tiếp đổi target của cohort từ hai hàng này. Nội dung HLS có thể ảnh hưởng nghiên cứu cognitive khác, nên không âm thầm deduplicate và không suy luận rằng toàn bộ feature pipeline bị đổi. Kinematic hiện tại lấy handwriting arrays, không lấy HLS làm predictor.
