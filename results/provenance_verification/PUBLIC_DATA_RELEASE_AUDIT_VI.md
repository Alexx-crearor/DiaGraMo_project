# Quyền công bố và privacy
Nguồn chính: [data descriptor](https://doi.org/10.1038/s41597-026-08625-x), [Zenodo v2](https://zenodo.org/records/21236910). Kiểm tra 2026-11-21T16:59:34.309386+00:00. Attribution: Zvončáková, Mekyska, Klocek, Mucha & Galáž (2026); dataset CC BY 4.0. Bảng kết quả local trỏ về Data Audit V3 tại commit f419530; không tái chạy V3.


**VERIFIED:** Zenodo API và data descriptor ghi CC BY4.0; không còn phù hợp ghi license dataset hoàn toàn unknown. **SUPPORTED:** consent mô tả chia sẻ tập biến pseudonymized cho nghiên cứu/công bố. Đây không phải bằng chứng mọi file local cũ, mọi bản gộp mới và mọi trường hợp liên kết đều được phép phát hành. Pseudonym không phải anonymization hoàn toàn.

**UNVERIFIED:** phiên bản chính xác của mọi local participant artifact và phạm vi cho derived individual OOF, SHAP, thumbnails, cognitive combinations. Công cụ pháp lý CC BY không tự chứng nhận rằng mọi liên kết có rủi ro riêng tư đã được đồng ý. Không đưa ra kết luận tư vấn pháp lý; đây là release-readiness audit.

Inventory 2666 tracked artifacts thuộc results/notebooks/docs/references; số theo phân loại: {'License status unknown': 67, 'Potentially sensitive participant-level': 1217, 'Public-safe only after review': 1378, 'Public-safe aggregate': 4}. Đây là conservative automated triage, không phải certify public-safe. Hình/binary/notebook outputs luôn cần review thủ công; không tự đánh dấu một file aggregate là an toàn chỉ bằng regex. Không scan blobs mọi commit; current tree và sự hiện diện trong HEAD đã đủ cho biết có lịch sử cần review. Không chứng nhận lịch sử sạch.

Đường dẫn ưu tiên: `results/data_audit_v3/audit_cohort.csv`, `participants_inventory.csv`, `label_disagreements.csv`, `omission_flag_details.csv`, `matched_oof_replay.csv`, `fusion_corrected_harmed.csv`; các bảng OOF trong `results/cv/`, XAI examples/waterfalls và notebooks có outputs. ID+city/site+tuổi/giới+cờ học tập có thể liên kết; individual probabilities/SHAP bổ sung suy luận mới. Không tái in hàng dữ liệu trong báo cáo này. `ARTIFACT_PRIVACY_CLASSIFICATION.csv` liệt kê đường dẫn, thay token participant trong tên file bằng `[participant-id]`; mapping đường dẫn nguyên vẹn chỉ lưu local.

**SUPPORTED — privacy v2:** ngày sinh/consent dates bị loại và administration dates làm thô. Local CSV headers phù hợp điểm này; không chứng minh toàn bộ RDS/JSON local không còn trường nhạy cảm. Cần đối chiếu release nguyên gốc có checksum và được chủ dữ liệu xác nhận trước khi tiếp tục phát hành row-level.

Không xóa file, không rewrite history, không đổi visibility và không push trong Phase 5. Kế hoạch remediation tách riêng; việc các dữ liệu đã ở GitHub không tạo thêm quyền công bố.
