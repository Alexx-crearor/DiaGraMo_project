# External validation requirements

**SUPPORTED — prospective design requirements, chưa phải kết quả external validation.**

| requirement | priority | acceptance | incompatibility_or_condition | verification_status |
| --- | --- | --- | --- | --- |
| x/y/time | Mandatory | Units, axes and timestamps must be documented and valid | Different units require predeclared conversion; no test-label tuning | SUPPORTED |
| Pen status | Mandatory | On/off-surface and transition conventions verified | Cannot reconstruct missing state without validated mapping | SUPPORTED |
| Pressure | Mandatory for exact frozen7 channel pipeline | Scale and calibration compatible | Missing pressure => not exact Kinematic/Temporal/fusion validation | SUPPORTED |
| Azimuth/tilt | Mandatory for exact frozen7 channel pipeline | Angle units/orientation/wrap verified | No silent zero fill | SUPPORTED |
| Timestamp units/sampling | Mandatory | Actual time, jitter/gaps and native/resampled convention | Unverified clock => no physical derivative comparability | SUPPORTED |
| Task identity/order | Mandatory for frozen 14-task models | Actual content, grade variants and mapping | Subset/different tasks => not exact frozen evaluation | SUPPORTED |
| Task instructions | Mandatory | Fast/precise, duration, language, stimuli permissions | No name-only compatibility assumption | SUPPORTED |
| Participant identity | Mandatory | Pseudonymized linkage and cohort independence | Repeated children must not cross partitions | SUPPORTED |
| Participant labels | Mandatory | Pre-existing independently adjudicated or explicitly screened target | Image-level labels cannot replace participant clinical labels | SUPPORTED |
| Diagnostic protocol | Mandatory | Instrument/rater/timing/ongoing status and clinical meaning | Unclear ascertainment => exploratory proxy target only | SUPPORTED |
| Population | Strongly preferred | Age/grade/language/education/case-mix documented | Mismatch requires transportability caveat | SUPPORTED |
| Site/device | Mandatory for source-independent claims | Both labels across multiple sources and device/operator logs | No source metadata => cannot verify source-independent validation | SUPPORTED |
| License/consent | Mandatory | Access, redistribution, derived outputs and attribution reviewed | Unknown rights => local access blocked pending permission | SUPPORTED |
| Cognitive scores | Optional | May contextualize labels but not enter frozen predictors | No added clinical claim from auxiliary scores | SUPPORTED |
| Offline image only | Not compatible | No valid temporal/multichannel mapping | Could support separate image study only | SUPPORTED |

## Bốn scenario

**A:** Cùng đủ7 channels và14 task, units/order/instructions được xác minh, labels/licensing đủ điều kiện: có thể đánh giá frozen Kinematic, Temporal và fixed50/50fusion sau kiểm tra schema/inference. Khóa weights, normalization, threshold và analysis plan trước nhận external outcomes. Báo cáo performance theo source, paired error overlap và uncertainty theo participant/site; không tune trên external Test.

**B:** Chỉ một số task trùng: chưa thể gọi là external validation của mô hình14 task. Chỉ dùng mô hình task-subset được phát triển/khóa trên development data trong nghiên cứu riêng được cho phép; cần đánh giá lại source Train và validation. Không chèn zero/mask rồi xem tương đương baseline.

**C:** Chỉx/y/time vàpen status: không chạy nguyên frozen7 channel pipeline như test hợp lệ. Có thể thiết kế nghiên cứu kinematic4channel riêng trước đó, nhưng phải retrain/validate trên development cohort theo protocol mới, ngoài phạm vi Phase 5. Không bịa pressure/angles.

**D:** Thiếu source metadata hoặc clinical codebook: chỉ kiểm tra technical inference khi schema đủ; chưa đủ chứng minh clinical/source-independent validation. Bổ sung quyền truy cập, ascertainment và source trước khi sử dụng làm xác nhận.

**UNVERIFIED — power:** chưa tính đủ cỡ mẫu cho external study vì chưa có mục tiêu sensitivity/specificity, margin và cấu trúc cluster chốt trước. Cần xác định endpoint, số site và số case/control trong mỗi site; tránh chọn sample size từ hiệu năng thuận lợi đã thấy.
