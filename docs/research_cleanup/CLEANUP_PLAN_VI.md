# Cleanup đã thực hiện theo hướng bảo toàn

1. Chụp Git/branch/inventory/hashes trước source edits; private exact paths chỉ local.
2. KEEP_ACTIVE: source4branches, Temporal/audit/native modules, configs/CLIs và notebook entrypoints. KEEP_HISTORICAL: mọi report, OOF, checkpoint, fold assignment, snapshot và nghiên cứu trước.
3. REVIEW_FOR_REMOVAL: cache sinh ra, tmp, môi trường cũ hoặc duplicate byte-identical; chỉ lập đề xuất. Không xóa bất cứ tệp nào.
4. Giữ đường dẫn artifact; thêm research index/registry/timeline, báo cáo tuần11slides và figure/claim review.
5. Refactor tối thiểu CLI/test scratch với tests và behavior checks; không rewrite pipeline, gộp reader hoặc đổi model.
6. Hash kiểm tra sau cùng, so sánh metrics/OOF, notebook/config và privacy; không retrain/commit/push/merge tự động.
