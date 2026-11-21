# RDS và truy vết raw → cleaned
Nguồn chính: [data descriptor](https://doi.org/10.1038/s41597-026-08625-x), [Zenodo v2](https://zenodo.org/records/21236910). Kiểm tra 2026-11-21T16:59:34.309386+00:00. Attribution: Zvončáková, Mekyska, Klocek, Mucha & Galáž (2026); dataset CC BY 4.0. Bảng kết quả local trỏ về Data Audit V3 tại commit f419530; không tái chạy V3.


**UNVERIFIED — RDS schema/attributes.** Hai RDS local được hash và kiểm tra dependency; decoded=False. `pyreadr`=False, `rdata`=False, Rscript=None. Không cài package, không thực thi R scripts làm sạch, không ghi vào nguồn. Chưa biết object type, factor levels, value labels hoặc attributes; CSV không thay thế bằng chứng đó. Dependency đề nghị nếu chủ project cho phép: pyreadr chính thức (https://github.com/ofajardo/pyreadr) trong virtualenv hoặc R chính thức; phải đọc RDS read-only và lưu schema/so sánh ở vùng local, không chạy pipeline gốc ghi đè.

**VERIFIED — nguồn thay thế sẵn có chưa được V3 khai thác:** raw CSV 289×762, cleaned CSV 289×768; cả hai có 288 unique IDs. Cờ tổng absent trong raw nhưng present trong cleaned. Cleaned có 13 flagged rows/13 unique IDs. Không đánh đồng số này với18/294 trong paper; kiểm tra release và RDS còn thiếu. Cohort bảo vệ có 3 IDs mang cờ.

**SUPPORTED — công thức:** OR của bốn component khớp cờ tổng ở 286/286 hàng complete; 3 hàng không đủ dữ liệu. Đây là tính tương đương quan sát trên bảng, không xác minh code tác giả hay NA semantics. Đối chiếu flags/labels cleaned với snapshot metadata V3 của258: True; có 0 IDs không unique trong nguồn CSV, được kiểm tra mọi giá trị thay vì chọn tùy ý.

**VERIFIED — tác giả cho phép lựa chọn giữ/loại:** bài gốc Usage Notes mô tả `keepOmittedParticipants_switch` (1giữ,0loại). Không có bằng chứng mặc định của script local và không tự áp dụng. Ba flagged người vẫn ở cohort vì `prepare.py` chỉ kiểm tra đủ14 task, không duplicate/invalid signal và binary `group`; không kiểm tra cờ omit. Do đó không thể gọi ba người này là vi phạm bắt buộc của tác giả chỉ từ tên cờ.

**UNVERIFIED — code nguồn:** paper nêu bốn R scripts00–03; project không chứa chúng và preview không truy được. Phải lấy bản không chứa dữ liệu cá nhân để xác minh exact formula, duplicate strategy và join cardinality.
