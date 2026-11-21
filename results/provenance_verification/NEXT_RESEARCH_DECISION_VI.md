# Quyết định nghiên cứu tiếp theo

## Case A — Provenance confirmed, vẫn confounded

**CONDITIONAL GO:** methodological comparison/ablation trong cùng cohort, có participant-independent folds cố định và giới hạn explicit. Có thể công bố empirical discrimination của dataset-defined target, cơ chế correction của fixed fusion, computational cost và nguy cơ source association. **NO-GO:** gọi superiority là clinical diagnosis/generalization, motor biomarker hoặc causal effect; không xem clarified labels là đã loại source confounding. Không cần chạy thêm architecture để giải quyết một câu hỏi về thiết kế mẫu.

## Case B — External balanced cohort available

**SUPPORTED — kế hoạch:** xác minh quyền, cohort independence, class×source overlap, schema/units/tasks và ascertainment. Khóa Kinematic/Temporal weights, Train preprocessing, thresholds và fixed50/50probability control trước nhận nhãn Test. Dùng blinded participant-level inference, paired errors và khoảng bất định, báo cáo từng source và class, tính cả specificity/calibration. Không chọn fusion weight bằng external Test. Nếu cần thích nghi task/channel thì tách development cohort/protocol mới và giữ một external lockbox độc lập.

## Case C — Không có dataset ngoài

**CONDITIONAL GO:** viết reproducibility/provenance paper hoặc methodological note về giới hạn target/source, exact OOF replay và compute trade-off; giữ mô hình hiện tại. Các phân tích city1 chỉ descriptive với7case, không làm nested search hoặc dùng uncertainty nhỏ giả tạo từ hàng triệu samples. Có thể thiết kế task-ablation hoặc source stress tests cho protocol tương lai, nhưng không tự triển khai trong Phase 5 và không đổi claim thành source independence. **NO-GO** cho clinical deployment/biomarker framing.

**Hành động ưu tiên duy nhất:** gửi tác giả gói câu hỏi về numeric target,5 discordances, versioned codebook/R scripts và quyền derived release qua owner. Nó giải quyết trực tiếp target và privacy, trong khi thêm training không thể xác định hai điều này. Đây là khuyến nghị, chưa gửi liên hệ.
