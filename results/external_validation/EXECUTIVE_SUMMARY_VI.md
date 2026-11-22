# Tóm tắt Phase 6

**Có raw Slovak thật, nhưng chưa đủ điều kiện đánh giá Kinematic/Stroke Transformer theo contract DiaGraMo.** Đã thực hiện tải nguồn, audit, đối chiếu nhãn, compatibility, adapter số, khóa protocol, đối chứng và kiểm thử. Chưa push/merge.

ZIP có121SVC cho120ID:120recording hợp lệ và một file0mẫu; mọi người vẫn có một recording hợp lệ. Tổng 2,176,419 samples trong bảy cột số. Excel có57DYSGR/63zero. HaiXML chứa192/155ID, không phải cohortcounts. Paper mô tả tám điều kiện viết nhưng không cung cấp task boundaries trong raw. Hai candidate time deltas âm được giữ nguyên. Thứ tự cột và physical units chưa được tác giả xác nhận.

**Đã chạy:** Train-prior label control,5/5participant folds,seed42, chỉ dùng class counts Train. Accuracy0,525; Macro-F1 0.344262, CI95% [0.302326; 0.381443]; balanced accuracy0,5, sensitivity0, specificity1. Đây là đối chứng class balance, không phải Kinematic hoặc Temporal. Pooled AUC≈0,48 do thay đổi prior giữa các folds, không phải motor information.

**Chưa chạy:** frozen transfer, shared-task transfer, raw Kinematic/compact Temporal và fusion/complementarity. Chúng bị chặn bởi semantic gate, không phải hết GPU budget. Prior fits cộng metrics/bootstrap dùng 11.110 giây wall/10.969 giây CPU; RSS lớn nhất quan sát 119.1MiB. GPU training bằng0.

Nguy cơ confounding vẫn còn: case/control đi qua các nguồn tuyển khác nhau; thiếu per-person source metadata để kiểm chứng held-out source. Public download không chứng minh quyền redistributing mọi raw/clinical/OOF artifact. Các hồ sơ cá nhân chỉ giữ trong vùng local đã Git-ignore.

**Quyết định:** dữ liệu thuộc classificationA (có raw), nhưng research OutcomeE cho signal-model execution với bằng chứng hiện tại. Phần điều tra có thể thực hiện đã hoàn tất; câu hỏi motor representation chưa giải quyết. Ưu tiên duy nhất: xin tác giả exact column schema/units và task/session mapping trước khi khóa một benchmark mới. Tests và hash checks thực xem VALIDATION_CHECKS.json; không coi NOT_RUN là PASS.
