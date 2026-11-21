# Phase 5 — Dataset Provenance & Clinical Label Verification

Nguồn chính: [data descriptor](https://doi.org/10.1038/s41597-026-08625-x), [Zenodo v2](https://zenodo.org/records/21236910). Kiểm tra 2026-11-21T16:59:34.309386+00:00. Attribution: Zvončáková, Mekyska, Klocek, Mucha & Galáž (2026); dataset CC BY 4.0. Bảng kết quả local trỏ về Data Audit V3 tại commit f419530; không tái chạy V3.


## 1. Executive Summary

**CONDITIONAL GO** cho nghiên cứu phương pháp; **NO-GO** cho khẳng định toàn bộ case đã được chẩn đoán Dysgraphia hoặc mô hình độc lập nguồn. Báo cáo kỹ thuật hoàn tất được với blocker rõ ràng; nguồn gốc lâm sàng chưa được xác minh đầy đủ. Không huấn luyện hoặc sửa kết quả cũ.

## 2. Original Dataset & Scientific Sources

Xem [nghiên cứu nguồn](ORIGINAL_DATASET_RESEARCH_VI.md) và [registry](SOURCE_DOCUMENTS.csv). Bài gốc đã có mục xác định nhãn và đạo đức. API nối được concept DOI với v2 và xác nhận CC BY 4.0. Chưa đối chiếu được checksum của bản dữ liệu local với archive v2; không tải archive chứa dữ liệu cá nhân.

## 3. Data Collection and Recruitment Provenance

[Acquisition matrix](ACQUISITION_PROVENANCE_MATRIX.csv) phân biệt tài liệu về tuyển mẫu, dòng thiết bị và bằng chứng cho từng phiên. **SUPPORTED:** khác biệt tuyển mẫu có thể góp phần tạo liên hệ nguồn–nhãn. **UNVERIFIED:** tác động riêng của mỗi nguồn, thiết bị hoặc người thu thập.

## 4. Definition and Validity of Labels

[Label provenance](LABEL_PROVENANCE_VI.md), [evidence matrix](LABEL_EVIDENCE_MATRIX.csv), [decision](LABEL_VERIFICATION_DECISION.json). Định nghĩa nhóm trong bài đã VERIFIED; ánh xạ mã số SUPPORTED; adjudication từng người UNVERIFIED. Cờ special-care không được xem là một chẩn đoán độc lập chỉ vì tên trường.

## 5. Five Disagreement Cases

Đã đối chiếu 5 trường hợp bằng cách chỉ đọc dữ liệu. Các trường liên quan khớp giữa raw CSV, cleaned CSV và merged JSON: True. 3 trường hợp tương thích với định nghĩa bao gồm ongoing diagnosis; hai trường hợp còn lại cần clarification. [Case decisions](DISAGREEMENT_CASE_DECISIONS.csv) không chứa ID hay clinical rows. Cả năm chưa được phân xử bằng hồ sơ cá nhân; không sửa nhãn.

## 6. Omission Flags and RDS Reconstruction

[RDS trace](RDS_SOURCE_TRACE_VI.md): decoded=False. Cleaned CSV có 289 hàng × 768 cột, gồm composite và component flags. OR của bốn component có 0 bất đồng trên 286 hàng đủ thông tin; không xem đây là bằng chứng về công thức R gốc. Bài báo cho phép người dùng chọn giữ/loại các trường hợp có cờ.

## 7. Duplicate Metadata and Merge Provenance

[Duplicate investigation](DUPLICATE_METADATA_INVESTIGATION.md): hai CSV local đều còn duplicate. Nội dung merged khớp occurrence thứ hai ở sáu trường HLS; điều đó không chứng minh quy tắc ưu tiên chính thức. ID này không thuộc cohort 258. Bất nhất với mô tả deduplication trong paper cần làm rõ theo phiên bản và code gốc.

## 8. Source–Label Confounding Interpretation

[Hypothesis matrix](HYPOTHESIS_EVIDENCE_MATRIX.csv), [interpretation](SOURCE_CONFOUNDING_INTERPRETATION_VI.md), [identifiability](IDENTIFIABILITY_DECISION.md). City-only accuracy 97,29% không chứng minh LightGBM dùng trực tiếp city/ID. Tài liệu tuyển mẫu hỗ trợ H1/H2, nhưng không định lượng được phần hiệu năng do nguồn hay vận động.

## 9. Acquisition & Protocol Unknowns

Paper mô tả Cintiq 16/Pro Pen 2, factory calibration, alignment và đơn vị seconds/mm. **SUPPORTED** ở mức tài liệu, chưa có log từng phiên. Average 167 Hz và cadence suy từ median khoảng 200 trong V3 là hai phép đo khác nhau, cần đối chiếu trước khi kết luận mâu thuẫn vật lý. 16 biến thể tên file được quy về 14 tác vụ do dictation/copy phụ thuộc khối lớp.

## 10. Public Data Release and Privacy Findings

[Release audit](PUBLIC_DATA_RELEASE_AUDIT_VI.md) và [remediation](REPOSITORY_REMEDIATION_PLAN.md) nêu các đường dẫn cần review. Giấy phép CC BY 4.0 được xác minh; phiên bản local và quyền phát hành derived OOF/SHAP chưa được xác minh đầy đủ. Inventory công khai che ID trong tên file; inventory nguyên vẹn giữ local. Không xóa hoặc sửa lịch sử Git.

## 11. External Validation Feasibility

[Requirements](EXTERNAL_VALIDATION_REQUIREMENTS.md), [shortlist](EXTERNAL_DATASET_SHORTLIST.md) có ba cohort online từ nghiên cứu gốc và một đối chứng loại trừ offline. Ưu tiên liên hệ tác giả cohort Slovak về schema/quyền dùng. Các cohort Pháp có hạn chế access, tác vụ và thiết bị. Chưa ứng viên nào đủ điều kiện xác nhận nguyên mô hình 14 tác vụ, 7 kênh.

## 12. Scientific Implications for LightGBM and Stroke Transformer

**VERIFIED theo V3:** Macro-F1 Kinematic khoảng 0,8472; Temporal 0,7132; fixed fusion 0,8407 trên target group. Việc làm rõ định nghĩa nhãn không biến các số này thành hiệu năng chẩn đoán đã xác nhận độc lập. Không chứng minh LightGBM chỉ học nguồn, cũng không chứng minh Transformer học motor biomarkers.

## 13. Verified vs Unverified Evidence

Bốn mức VERIFIED/SUPPORTED/UNVERIFIED/CONTRADICTORY áp dụng theo đúng phạm vi claim. Định nghĩa công bố không thay thế adjudication từng người; mô tả code không thay thế kiểm tra source code; giấy phép không tự chứng minh mọi liên kết dữ liệu đều được chấp thuận. Xem các matrix, decision JSON và blockers.

## 14. Required Author/Owner Clarifications

[15 câu hỏi tiếng Anh](AUTHOR_QUESTIONS_VI.md) ưu tiên mã hóa nhãn, ongoing cases, năm bất đồng, R scripts, omission/duplicate policy, phiên bản và quyền phát hành. Chưa gửi liên hệ; không đưa ID hoặc clinical rows vào trao đổi công khai.

## 15. Research GO / CONDITIONAL GO / NO-GO Decision

[Next research decision](NEXT_RESEARCH_DECISION_VI.md): CONDITIONAL GO cho nghiên cứu phương pháp và tái lập có giới hạn minh bạch; NO-GO cho triển khai lâm sàng hoặc claim độc lập nguồn. Ưu tiên xin codebook, R scripts theo phiên bản và clarification nhãn, trước khi nghiên cứu thêm kiến trúc.

## 16. Reproducibility and Limitations

[Reproduce](REPRODUCE.md), [validation](VALIDATION_CHECKS.json), [manifest](ARTIFACT_MANIFEST.json), [blockers](BLOCKERS.md). Kiểm tra hash toàn bộ tracked files tại mốc và các protected inputs đã chọn. Không tái chạy sensor audit V3, không train, không tải dataset, không push. Chưa quét toàn bộ Git history về privacy; RDS vẫn thiếu reader.
