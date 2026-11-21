# Phase F — Validation feasibility và quyết định

| source_field | sources | sources_with_two_classes | sources_allowing_both_train_test_classes | largest_within_source_minority |
| --- | --- | --- | --- | --- |
| date_of_administration | 7 | 6 | 6 | 18 |
| id_city | 15 | 1 | 0 | 7 |
| id_school_ppp | 35 | 4 | 4 | 2 |
| site_type_ppp | 2 | 1 | 0 | 89 |

- **A — Original six-fold:** đánh giá khả năng dự đoán group cho người mới trong hỗn hợp nguồn hiện tại, với source overlap. Không chứng minh generalization nguồn mới, chẩn đoán lâm sàng hay model selection độc lập.
- **B — Within-source:** city1 có hai lớp nhưng minority nhỏ; với6fold chỉ1–2 positives/Test nên sensitivity nhảy50–100 điểm% theo một người. Không trình bày như nghiên cứu đủ lực thống kê. Một số school/site có hai lớp nhưng cũng rất ít minority; metadata population/protocol chưa cân bằng.
- **C — Leave-One-Source-Out:** city1 Test có hai lớp nhưng Train thiếu lớp0; city khác Test chỉ có một lớp. Site/date held-out có thể khả thi theo đại số ở vài nguồn, chưa đủ support và có thể vẫn overlap city/hardware. Không tự chạy một CV mới hoặc gọi nó là giải pháp confounding.
- **D — External validation:** cần cohort độc lập theo recruitment/site/time và decision process; reference labels có định nghĩa/công cụ/người đánh giá, omission policies,14-task compatibility, channels/timestamp units, device/firmware/export version, calibration/SOP/operator, missingness và harmonization plan Train-only được khóa trước Test. Cần cả hai lớp ở Test và fit population phù hợp.
- **E — Multi-source balanced collection:** thu cả hai lớp ở nhiều nguồn, có clinical ascertainment tương đương và điều kiện acquisition được ghi nhận; tránh source gần như đồng nhất label. Dự tính sample size từ độ chính xác mong muốn của sensitivity/specificity và paired improvement, tính clustering theo site, không chọn một con số tùy tiện. Có thể bổ sung crossover thiết bị hoặc standardized reference traces để phân biệt acquisition và population.

| direction | decision | prerequisite |
| --- | --- | --- |
| Temporal encoder expansion on current cohort | NO-GO for confirmatory claims; CONDITIONAL GO for explicitly exploratory methodological controls | Resolve labels/source provenance and independent balanced validation; no architecture race now |
| Train-only calibrated fusion | CONDITIONAL GO as separately preregistered exploratory control, not clinical validation | Train-only calibration/cross-fitting and independent/source-aware evaluation; current OOF cannot select parameters |
| SSL | NO-GO as current priority | SSL may preserve/amplify source information; first establish source/clinical identifiability |
| Static Visual branch | NO-GO as current priority | Images can also encode source/protocol; require data validity and a separately justified hypothesis |
| Source invariance / cross-domain / clinical biomarkers | NO-GO claims | Independent multi-source balanced data, verified reference labels and acquisition metadata |

**Quyết định tổng thể: CONDITIONAL GO cho nghiên cứu phương pháp thăm dò có giới hạn; NO-GO cho confirmatory/clinical/source-invariant claims trên thiết kế hiện tại.** Ưu tiên data/codebook/provenance và independent balanced validation; không tự tăng Transformer, thêm nhánh ảnh, SSL hoặc tìm fusion để chạy theo F1. Không kết luận LightGBM/Transformer đã học sai; hiện không định lượng được bao nhiêu phần performance đến từ source shortcut thay vì motor evidence.
