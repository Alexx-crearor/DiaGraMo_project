# Đề xuất review — chưa xóa

`DUPLICATE_ARTIFACT_REVIEW.csv` có paths an toàn/alias, size, hash, reason và dependency warning. Exact private paths ở ignored inventory để chủ sở hữu review local.

- Byte-identical historical reports/config snapshots/checkpoints: **KEEP_HISTORICAL mặc định**; bản sao có thể chứng minh provenance theo phase, không tự deduplicate.
- `__pycache__`, `.pytest_cache`, `tmp`: REVIEW_FOR_REMOVAL theo inventory; có thể tái tạo nhưng phải xác minh không còn job/lock phụ thuộc và xin owner approval trước.
- `.venv314`: chỉ review môi trường cũ sau khi xác minh kernel/notebook/automation không dùng; không xóa trong task này.
- Hai requirements files không duplicate semantics. `.cache` pretrained weights, embedding caches và raw recordings luôn giữ để tránh tải/trích xuất lại.
- Bốn tệp untracked có trước cleanup, gồm review note và completion watcher artifacts: giữ nguyên byte; không stage/xóa/di chuyển.

Điều kiện trước bất kỳ removal tương lai: explicit owner approval cho danh sách cụ thể, dependency/use check, và manifest/Git refs còn truy xuất được. Không có deletion approval trong task này.
