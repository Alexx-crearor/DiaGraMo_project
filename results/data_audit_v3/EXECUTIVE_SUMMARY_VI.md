# Executive Summary — DiaGraMo Data Audit V3

## Kết luận chính

**CONDITIONAL GO** cho nghiên cứu phương pháp thăm dò; **NO-GO** cho tuyên bố clinical biomarkers, source-invariance hay cross-domain generalization từ cohort hiện tại. Chưa có cơ sở ưu tiên tăng độ phức tạp mô hình. Không chứng minh mô hình đã học sai; chỉ xác định các giới hạn của dữ liệu và phép đánh giá.

## Các verified facts và associations quan trọng

1. Tái tạo đúng 258 IDs/labels/folds; 294 merged files và 288 SVC directory IDs không phải cùng một inventory. Có 12 source-link anomalies ngoài cohort, không tự sửa danh tính.
2. Group và specific dysgraphia flag khác ở 5 người; 3 người mang omit flag vẫn được giữ theo baseline. Raw CSV thiếu total omission flag; clinical label definitions/flag-generation provenance chưa xác nhận. Duplicate cognitive ID có các trường HLS xung đột.
3. City1: 115 group0/7 group1; city khác: 0/136. City–group V=0.946827. Mapping city chỉ học Train đạt OOF accuracy=0.972868, prior=0.554264. Đây là diagnostic source→dataset label, không clinical classifier.
4. Source overlap làm participant-independent CV không trở thành source-independent. City-held-out không có split vừa Train vừa Test đủ hai lớp. Within-city1 chỉ ít positives nên không có precision đáng tin để khái quát rộng.
5. Raw SVC/cached21channels/statistics được kiểm tra toàn cohort; preprocessing/source differences được lượng hóa ở PhaseD, không suy hardware causation. Units thiết bị và clinical codebook chưa xác minh.
6. Full OOF giữ kết quả cũ: Kinematic tốt hơn fixed average về point Macro-F1; fusion corrected=20, harmed=22, ΔF1=-0.006483, CI95%[-0.053470,+0.041286]. Source/subgroup/sensitivity đều hậu nghiệm; không chọn winner.

| model | accuracy | f1_macro | auc | brier |
| --- | --- | --- | --- | --- |
| Kinematic_LightGBM | 0.848837 | 0.847163 | 0.906902 | 0.122450 |
| Stroke_Temporal | 0.713178 | 0.713178 | 0.807540 | 0.224193 |
| Fixed50 | 0.841085 | 0.840681 | 0.906111 | 0.129793 |

## Việc cần làm trước mô hình tiếp theo

Xin tác giả/chủ dữ liệu codebook clinical, cohort list/omit rules, lý do disagreement, pipeline CSV/RDS→merged và duplicate-ID resolution; device/clock/export/operator/SOP/recruitment records. Thiết kế dữ liệu có cả hai lớp tại nhiều nguồn và test độc lập. C1 structured-temporal mechanism cần equal-capacity/equal-budget controls và OOD-aware perturbations riêng; C2 complementary motor evidence cần calibrated Kinematic control Train-only và validation độc lập. Chưa chạy thêm thí nghiệm.

Direct leakage, source confounding và researcher-level selection bias là ba vấn đề khác nhau. Audit code/fit-ID provenance không phát hiện đường trực tiếp đưa source ID/cognitive label vào features, nhưng không thể chứng minh vắng mọi leakage hay phân rã motor/source causally. Các giả thuyết population, protocol và thiết bị còn unidentifiable.

Code mới `scripts/data_audit_v3/`, tests riêng `tests/data_audit_v3/`, mọi outputs `results/data_audit_v3/`. Không thay đổi input/cohort/model/folds/checkpoints/OOF cũ; xem `VALIDATION_CHECKS.json` và manifest để biết phạm vi xác minh cuối cùng. Không push hoặc merge.
