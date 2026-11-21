# Blockers và giới hạn không thể tự giải quyết

- Không thiếu SVC/cache/OOF cần thiết cho cohort hiện tại nếu D_STATUS và EF_STATUS complete.
- Thiếu local clinical codebook, recruitment SOP, official cohort/omit derivation và thiết bị/units/calibration metadata. Không thể xác minh clinical semantics hoặc nguyên nhân confounding; cần chủ dữ liệu.
- Hai RDS hiện diện: `data/raw_data/cognitive_test_data/cognitiveTestRdata.rds` và `data/processed_data/cognitive_test_data/cognitiveTestRdata_cleaned.rds`. Không có local pyreadr/Rscript; chỉ inventory/hash, chưa đọc attributes/value labels. CSV/merged vẫn đủ cho các phép đối chiếu nhị phân và counts đã công bố. Không cài/tải qua mạng.
- Source-link anomalies ngoài cohort và duplicate clinical ID cần giải thích, không tự nối/sửa.
- Participant OOF không chứa task-specific attributions; không xác định task gây correction. Không chạy inference ablation mới.
- Working tree có4 untracked files cũ; điều kiện clean-tree để tạo nhánh audit chưa thỏa. Giữ nguyên nhánh, không stash/xóa/commit/push.
