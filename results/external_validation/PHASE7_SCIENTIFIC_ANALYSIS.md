# Phase 7 — Phân tích khoa học và negative controls

## Bằng chứng thực sự

Raw có thật, gồm120recording hợp lệ và một file0mẫu. Đã join đủ120ID/labels, giữ các recording của cùng người trong cùng candidate fold. Không có bảng engineered features phát hành trong3files nguồn đã tải. Không có task labels để tách8điều kiện viết được mô tả trong bài gốc.

Chỉ có prior-control OOF; không có kết quả Kinematic/Temporal Slovak. Do đó không thể kết luận Temporal tốt/kém hơn, không tính complementarity, corrected/harmed hoặc fusion. `complementarity_metrics.csv` ghi NOT_COMPUTED và paired count0, không điền số0 giả cho các đại lượng chưa tính. Không so sánh prior-control Slovak với0.8472F1 của DiaGraMo như paired test.

| model | participant_count | completed_folds | metric | value | ci95_low | ci95_high | fold_mean | fold_sd | evaluation_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train_prior_label_control | 120 | 5 | accuracy | 0.525000 | 0.433333 | 0.616667 | 0.525000 | 0.022822 | Exploratory external-dataset-only internal CV; label-only control |
| Train_prior_label_control | 120 | 5 | f1_macro | 0.344262 | 0.302326 | 0.381443 | 0.344144 | 0.009869 | Exploratory external-dataset-only internal CV; label-only control |
| Train_prior_label_control | 120 | 5 | auc | 0.479950 | 0.394880 | 0.565175 | 0.500000 | 0.000000 | Exploratory external-dataset-only internal CV; label-only control |
| Train_prior_label_control | 120 | 5 | balanced_accuracy | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.000000 | Exploratory external-dataset-only internal CV; label-only control |
| Train_prior_label_control | 120 | 5 | sensitivity | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | Exploratory external-dataset-only internal CV; label-only control |
| Train_prior_label_control | 120 | 5 | specificity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | Exploratory external-dataset-only internal CV; label-only control |
| Train_prior_label_control | 120 | 5 | brier | 0.249609 | 0.245068 | 0.254233 | 0.249609 | 0.001248 | Exploratory external-dataset-only internal CV; label-only control |
| Train_prior_label_control | 120 | 5 | log_loss | 0.692367 | 0.683275 | 0.701624 | 0.692367 | 0.002499 | Exploratory external-dataset-only internal CV; label-only control |
| Train_prior_label_control | 120 | 5 | ece10 | 0.000000 | 0.000174 | 0.100608 | 0.025000 | 0.005705 | Exploratory external-dataset-only internal CV; label-only control |

Pooled prior AUC≈0.48 trong khi mỗifold AUC0.5: Trainprior thay đổi đối ứng với tỷ lệ nhãn heldoutfold. Đây là hiệu ứng thiết kế của control, không phải học tín hiệu âm. ECE10=0 do các xác suất cùngbin và meanprior khớp tỷ lệ nhãn tổng; điều đó không chứng minh một mô hình lâm sàng được calibration tốt. Sensitivity0 và specificity1 phản ánh luôn dự đoán nhóm0.

CI95% là percentile2000lần resample các participant OOF với xác suất cố định. Không resample modeltraining; chưa xử lý siteclustering vì thiếu source metadata. Một số CI control suy biến vì cùng dự đoán một lớp; không xem là độ chắc chắn về hiệu năng trên nguồn mới.

## Domain shift và source

`domain_shift_summary.csv` tách quan sát native/candidate với đại lượng vật lý. Median c2delta8 không tự có nghĩa8ms/125Hz; so trực tiếp với0.005source-time của DiaGraMo là không hợp lệ. Không quy mọi khác biệt pressure/góc/duration cho hardware. Không gán nghĩa motor sinh học cho cột chưa được xác nhận.

Original Methods mô tả tuyển case ở cơ sở special-needs, controls ở trường; dataset thứ hai vì vậy không tự giải quyết source confounding. Source×class table ở từngngười không có; không tạo source bằng cách sao label rồi báo perfectnegativecontrol. `source_confounding_summary.csv` phân biệt V3observations với documentary risk ởSlovak.

## Quyết định

**NO-GO hiện tại cho signal-model transfer/comparison**; **CONDITIONAL GO** sau khi tác giả xác nhận exactexport schema/units và task/sessionmapping. Không có thất bại performance Temporal để giải thích: nó chưa chạy, nên không đổ cho inadequatelearning/nhỏN. Đây là thiếu điều kiện thực nghiệm, không phải một kết quả mô hình âm.
