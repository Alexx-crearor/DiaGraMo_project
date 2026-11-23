# Bàn giao cleanup và chuẩn bị báo cáo tuần

## 1. Tóm tắt

Trạng thái: **COMPLETE**. Đã tổ chức lại điểm vào và tài liệu, không tạo experimental phase hoặc chạy research training. Nhánh `study/weekly-research-cleanup`, base `abee084d6621ccf4175f3e4a1a5a484756bb824f`.

## 2. Cấu trúc ban đầu

82,441 files, 41.240GiB ngoài Git internals; 3003 tracked. Data/cache/weights/results và hai Python environments chiếm phần lớn dung lượng. Snapshot hash 19,493 files/34.701GiB; môi trường Python và generated pycache/pytest cache được inventory nhưng không content-hash. Exact private paths chỉ ignored local.

## 3. Tổ chức cập nhật

`docs/research/`: registry/timeline/architecture/results/limitations. `docs/weekly_reports/2026-11-16_to_2026-11-22/`: báo cáo,11slides, figures/claims/next-week plan. `docs/research_cleanup/`: inventory/dependencies/duplicates/checks. Không di chuyển result directories hoặc đổi CLI paths.

## 4. Files bổ sung

Scripts audit/build_registry/documents/checks/safe_tests/validate/finish, cleanup regression tests, research/weekly documentation và manifests. Inventory số từng nhóm trong PROJECT_INVENTORY.csv; registry 345 records. Guide README lịch sử giữ riêng để old instructions còn truy cập được.

## 5. Files sửa

README.md, scripts/native_channel_research/run.py, tests/native_channel_research/test_native.py.

README được rút gọn thành entrypoint; source chỉ đổi imported CLI argv/NumPy scope và test scratch fixture. Không sửa bất cứ file src/diagramo nào.

## 6. Files chủ động bảo toàn

Mọi raw SVC, merged/cleaned data, embedding cache, pretrained weights, model checkpoints, OOF, fold assignment, configs, notebooks01–09, baseline4branches, XAI/Temporal/audit/provenance/external/native reports. Giữ cả bốn pre-existing user untracked files. Không chạm history hoặc legacy removal plan.

## 7. Refactor thực hiện

Xem CODE_REFACTOR_SUMMARY.md: imported CLI giữ default command behavior; pytest scratch isolation tránh overwrite checkpoint test cũ. Caller map/AST và CLI help equivalence được lưu. Không gộp reader/metrics khác scientific semantics.

## 8. Duplicate đã nhận diện

1100 duplicate-member entries thuộc 490 exact-byte groups; đây là bằng nhau về byte, không chứng minh dư thừa về provenance. Những snapshot/weights/cache có thể có chủ đích. PROJECT_INVENTORY.csv cũng đánh dấu generated caches theo policy.

## 9. Removal đề xuất, chưa thực hiện

REMOVAL_REVIEW_PLAN.md và DUPLICATE_ARTIFACT_REVIEW.csv chứa path/alias, size, reason, dependency warning. REVIEW_FOR_REMOVAL cần owner approval và kiểm tra use trước khi xóa. Không có file nào bị xóa, không dọn `.venv314`, tmp hoặc caches.

## 10. Kết quả khoa học bảo toàn

Kinematic F1=0,847163; Temporal=0,713178; Fixed50=0,840681. Fusion ΔF1≈−0,006483, CI95%[−0,053470;0,041286]. City-only accuracy≈0,972868. Slovak B0–B4 F1 lần lượt0,344262/0,598214/0,932413/0,941337/0,941337. B3/B4 binary predictions giống nhau; B3−B2 CI chứa0. Tất cả đọc/replay từ artifacts cũ, không tính score cho model mới. XAI tables, confidence intervals và source statistics nằm trong hash gate.

## 11. Tests và checks đã chạy

| Stage | Suites | Passed tests | Failed suites |
|---|---:|---:|---|
| before | 8 | 165 | none |
| after | 9 | 172 | none |


Logs chi tiết chỉ local, public summary không in clinical IDs. BEFORE gồm165tests PASS. AFTER bỏ đúng một historical full-tree guard do source/README edits hợp lệ; guard đã PASS trước sửa, được thay bằng new full preservation manifest. Không gọi full runners để test. Unit tests có synthetic forward/backward/optimizer steps; không train research model hoặc tạo performance result mới. Xem TEST_RESULTS.json cho status chính xác; CLI help, syntax, documentation links và OOF replay có JSON riêng.

## 12. Protected artifact integrity

PRESERVATION_VALIDATION.json: True. PROTECTED_BEFORE/AFTER ghi counts/hash status, exact local manifest giữ private. Allowed changed files chỉ README/native CLI/native tests; mọi result/input/checkpoint phải giữ nguyên SHA. Không coi count-only check là content preservation.

## 13. Riêng tư

Không thêm data/weights/checkpoints/individual clinical rows/OOF vào Git. Public inventory gộp sensitive roots và che sensitive filenames; known-ID token scan và output selection có trong PRIVACY_REVIEW.json. Không chứng nhận de-identification toàn Git history. Historical artifacts có risk được bảo toàn, không tự xóa/redistribute.

## 14. Sẵn sàng báo cáo tuần

Báo cáo10mục,11slide entries, figure/table index và12claim checks hoàn tất nội dung. Ba aggregate figures đã được xem trực quan. Dùng bảng gốc cho locked/native comparisons, không vẽ lại hàng loạt. XAI là previous milestone; diagram mô tả đúng relative segment attention/task pooling. Không tạo PowerPoint trong task này.

## 15. Technical debt còn lại

Historical scripts có nhiều global imports, module report.py trùng tên và format compact. Dynamic imports không được AST chứng minh hoàn toàn; giữ explicit namespace và suite isolation. Một số CLI cũ không có --help an toàn (provenance runner), không gọi thử tùy tiện. Requirements baseline/XAI bổ sung khác vai trò, không auto merge. Historical full-tree hash guards cần cơ chế baseline ownership cho thay đổi documentation tương lai; không sửa snapshot cũ trong cleanup.

## 16. Bước phát triển đề xuất

Ưu tiên source-aware replication và một protocol development-only để tách marginal/implicit-position/order hypotheses. Không tăng Transformer hoặc full SSL mặc định. Owner cần quyết định removal cụ thể, release privacy cho hình nhóm nhỏ, clinical/source clarification và access cohort mới. Không tự gửi liên hệ hoặc push/merge.
