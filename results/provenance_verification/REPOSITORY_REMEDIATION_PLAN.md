# Repository remediation plan — cần owner review

**SUPPORTED — hành động ưu tiên:** tạm dừng phát hành participant-level artifacts mới. Giữ nguyên source và lịch sử để có thể điều tra; Phase 5 chỉ thêm báo cáo aggregate và code. Vùng `data/provenance_verification_local/` đã được rule `data/` hiện có ignore, không cần sửa .gitignore toàn project.

1. Owner đối chiếu exact dataset version/checksum, CC BY attribution và consented fields với tác giả. Xác định derived OOF/SHAP có thuộc phạm vi phát hành được chấp thuận hay không.
2. Review inventory theo đường dẫn, kể cả embedded notebook/PNG/SVG và các case nhỏ có thể liên kết. Chuyển public report sang aggregate; không dùng ID hash cố định như một bảo đảm anonymization. Suppress small cells khi cần, giảm precision tuổi/địa điểm/ngày, bỏ case-level values trong public outputs. Đánh giá khả năng liên kết qua các file cũ, không chỉ từng file đơn.
3. Chỉ sau owner quyết định mới lập patch thay public exports, đưa row-level mới vào vùng ignored. Giữ evidence read-only tại máy có kiểm soát truy cập. Không thay đổi số liệu khoa học/baseline khi tạo bản aggregate.
4. Nếu phát hành trước đó vượt phạm vi được phép, xóa khỏi HEAD là chưa đủ. Cần owner phối hợp tác giả/GitHub quản lý history/forks/cache; review paths và backup trước mọi rewrite. Hiện chưa có kết luận pháp lý bắt buộc rewrite, nhưng history review là cần thiết.

Các lệnh tham khảo **chưa chạy**, chỉ phục vụ review:

```powershell
git ls-files results notebooks
git log --all -- results/data_audit_v3/audit_cohort.csv
git check-ignore data/provenance_verification_local/UNRESOLVED_LABEL_CASES.csv
```

Nếu sau này dùng công cụ history filtering, phải dùng danh sách path đã duyệt trong bản clone backup và có phê duyệt owner. Không cung cấp/chạy lệnh force-push trên working repository này. **UNVERIFIED:** phạm vi quyền cho từng derived artifact; không thể suy ra từ public download.
