# Tái lập Phase 5

Chạy từ repository root trên virtualenv hiện tại; không chạy prepare.py, scripts/report.py hoặc R cleaning scripts.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/provenance_verification/run.py all
```

Các phase riêng: `analyze`, `reports`, `tests`, `validate`. Snapshot bảo vệ tạo một lần và giữ trong `data/provenance_verification_local/PROTECTED_BEFORE.json`; rerun không thay mốc. Code không fit model, không network-fetch, chỉ đọc source local và ghi hai folder mới dành riêng. Explicit file-path module imports ngăn va chạm report cũ.

Nguồn web đã đọc thủ công theo SOURCE_DOCUMENTS.csv; cached PDF/API/README chỉ ở vùng dataignored, không được tự fetch lại hoặc coi cached author-document hash là dataset-file hash. Registry ghi thời gian kiểm tra, khả năng truy cập và SHA khi có. PDF chính đọc bằng pypdf; PDF Asselborn dùng PyMuPDF do pypdf tokenlimit. Downloaded content chỉ gồm tài liệu/paper/metadata, không participantdataset.

Nếu copy repo sang máy khác, cần source local có quyền dùng và cached tài liệu hoặc kiểm tra lại primary URLs trước khi giữ các claim về nguồn. Evidence synthesis là bản review bằng tay gắn nguồn, không tự suy luận clinical definition từ column names. RDS absentdependency giữ blocker; nếu reader mới có sẵn phải kiểm tra schema/attributes và review report trước khi thay trạng thái.

Tests kiểm tra join không mơ hồ, prefixNA, missingness, đường dẫn output, imports, privacy guard, case decisions và RDS schema hoặc blocker. Validation hash toàn bộ tracked files tại mốc, các CSV/RDS và merged case files đã đọc, derivedcohort/features. Không quét lại4000SVC hoặc train. Tests/validation/manifests tạo kết quả riêng; manifest không self-hash. No commit/push.
