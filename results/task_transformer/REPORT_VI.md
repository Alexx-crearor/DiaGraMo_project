# Motor-Invariant Kinematic Task Transformer — báo cáo thực nghiệm

Trạng thái số liệu: hoàn tất huấn luyện và phân tích sáu outer folds; chỉ coi toàn bộ deliverables hoàn tất khi `validation.json` có `complete: true`. “Motor-invariant” là **giả thuyết nghiên cứu**, chưa phải thuộc tính đã chứng minh. Nghiên cứu chỉ dùng Kinematic, không triển khai Fusion hay cross-language.

## Kết quả chính

Baseline Kinematic + LightGBM được giữ nguyên: Accuracy OOF **84.8837%**, macro-F1 **0.8472**, ROC-AUC **0.9069**. Bộ chọn cấu hình neural hoàn toàn bằng inner CV (`nested_selected`) đạt Accuracy **72.4806%**, macro-F1 **0.7208**, AUC **0.7731**; chênh lệch macro-F1 OOF so với baseline **-0.1264**. Đây là kết quả của quy trình chọn họ mô hình trong Train, không phải chọn họ có outer Test cao nhất sau khi xem bảng.

Chưa có bằng chứng để thay thế LightGBM bằng Transformer: nested neural thấp hơn baseline về macro-F1 trên bộ dữ liệu này.

| Mô hình | f1_macro OOF | f1_macro fold | accuracy OOF | accuracy fold | auc OOF | auc fold | balanced_accuracy OOF | balanced_accuracy fold |
|---|---|---|---|---|---|---|---|---|
| lightgbm | 0.8472 | 0.8470 ± 0.0324 | 0.8488 | 0.8488 ± 0.0321 | 0.9069 | 0.9040 ± 0.0313 | 0.8475 | 0.8473 ± 0.0323 |
| lightgbm_without_text | 0.6930 | 0.6917 ± 0.0412 | 0.6977 | 0.6977 ± 0.0416 | 0.7697 | 0.7690 ± 0.0568 | 0.6924 | 0.6926 ± 0.0417 |
| task_mlp | 0.7458 | 0.7440 ± 0.0479 | 0.7519 | 0.7519 ± 0.0433 | 0.8203 | 0.8283 ± 0.0468 | 0.7439 | 0.7439 ± 0.0466 |
| transformer | 0.7385 | 0.7359 ± 0.0388 | 0.7442 | 0.7442 ± 0.0360 | 0.8157 | 0.8284 ± 0.0360 | 0.7369 | 0.7378 ± 0.0382 |
| motor_normalized | 0.7239 | 0.7172 ± 0.0895 | 0.7287 | 0.7287 ± 0.0815 | 0.7985 | 0.8361 ± 0.0737 | 0.7229 | 0.7229 ± 0.0895 |
| xai_selected | 0.7680 | 0.7627 ± 0.0782 | 0.7713 | 0.7713 ± 0.0742 | 0.8317 | 0.8518 ± 0.0724 | 0.7673 | 0.7682 ± 0.0723 |
| train_augmented | 0.7411 | 0.7374 ± 0.0705 | 0.7442 | 0.7442 ± 0.0674 | 0.7987 | 0.8145 ± 0.0529 | 0.7411 | 0.7417 ± 0.0688 |
| without_text_tasks | 0.7018 | 0.7006 ± 0.0496 | 0.7093 | 0.7093 ± 0.0504 | 0.7642 | 0.7821 ± 0.0542 | 0.7003 | 0.7011 ± 0.0507 |
| nested_selected | 0.7208 | 0.7150 ± 0.0274 | 0.7248 | 0.7248 ± 0.0229 | 0.7731 | 0.8007 ± 0.0383 | 0.7202 | 0.7213 ± 0.0242 |

OOF gộp 258 người khác với trung bình sáu fold; SD theo sáu fold dùng ddof=1, không phải khoảng tin cậy. Chi tiết toàn bộ đối chứng trong `tables/ablation_primary.csv`; ba seeds trong `tables/seed_sensitivity.csv`.

## Kiểm toán và dữ liệu

Làm việc trên branch `study/kinematic-task-transformer`, kế thừa project hiện có. Tái sử dụng nguyên extractor `diagramo.features.kinematic`, cache `data/derived/kinematic.npz`, mapping `schema.json`, fold assignments và checkpoint LightGBM cũ. 258 participants, nhãn `group` (115/143), 14 tác vụ × 74 đặc trưng = 1.036 chiều. Mapping thực tế ở `feature_mapping.csv`; không tự sắp xếp tác vụ theo tên hoặc giả định trình tự thời gian.

Đã tái trích xuất nguyên bản cho toàn bộ 258 người và kiểm tra **khớp chính xác** cache cũ. 19 banks biến đổi không dùng nhãn, 21,672 kiểm tra raw-record/condition; các JSON nguồn có SHA256. Artifact lịch sử được bảo vệ bằng `protected_manifest.json` và xác minh lại lúc validation. Task 1 và pipeline bốn nhánh không bị sửa.

## Kiến trúc và giao thức tránh leakage

Mỗi task là một token 74 chiều. Median imputation giữ đủ 74 tọa độ và StandardScaler chỉ fit trên Train tương ứng. Shared Linear 74→32/64 + GELU + dropout; optional task-ID embedding; 1–2 pre-LayerNorm self-attention blocks, 2/4 heads, FFN chiều 2d; mean hoặc learned attention pooling; LayerNorm và binary logit. Không có chronological positional encoding. Task-MLP dùng shared token feed-forward blocks và mean pooling, không có attention. Missing tokens bị triệt tiêu ở input, attention keys và pooling. Các mạng được train với task dropout 0,1, bảo đảm còn ít nhất một task.

Quy trình cố định outer Stratified 6-Fold seed 42, mỗi Test 43 người, Train 215. Inner Stratified 3-Fold chỉ trong outer Train. Mỗi family có bốn ứng viên cố định (`configs/task_transformer.json`): d32/no task-ID/mean; d32/task-ID/mean; d64/task-ID/mean; d32/hai layers/task-ID/attention pool. Task-MLP luôn mean pooling. Tối đa 50 epochs, patience 7, chạy ít nhất 10; AdamW lr 0,001, weight decay 0,001, batch64, gradient clipping1. Chọn checkpoint theo inner validation BCE; chọn cấu hình theo mean inner macro-F1, tie-break AUC, ít tham số, thứ tự cấu hình. Final model dùng toàn bộ outer Train và số epochs là median best epoch của ba inner runs; **không early-stop trên outer Test**.

432 inner neural fits; 144 final neural fits gồm 108 fits của sáu families × ba seeds × sáu folds và 36 matched controls seed42. 36 tree model records gồm sáu checkpoint baseline tái sử dụng và 30 fits mới (hai seed bổ sung/full tasks, ba seed/without text). XAI task selection tái cross-fit ba lần trong từng inner Train: 54 LGBM fits bổ sung; final outer Train chỉ tái sử dụng Task-1 ranking khi IDs và hash đầu vào khớp. Sáu LGBM fits khác chỉ đo thời gian và phải khớp dự đoán baseline, không dùng chọn mô hình. Pilot 10 epochs chỉ trên outer Train0 để kiểm tra tài nguyên, không dùng Test chọn grid.

Rank top-k=7 hoặc10 chỉ từ Train; trong grid, top-k còn đi cùng thay đổi kiến trúc nên không suy diễn tác động riêng top-k từ grid. Có matched top10/full14 cùng cấu hình để hỗ trợ kiểm chứng. Ba seeds 42/43/44 giữ cấu hình được chọn bằng seed42 và fold assignments; đây là sensitivity của huấn luyện, không phải 18 folds độc lập hoặc ba cohorts.

Các lựa chọn cuối cùng của nested neural:

| fold | source_model | candidate |
|---|---|---|
| 0 | transformer | 0 |
| 1 | train_augmented | 2 |
| 2 | xai_selected | 2 |
| 3 | task_mlp | 0 |
| 4 | train_augmented | 0 |
| 5 | xai_selected | 3 |

**Giới hạn cấp nghiên cứu:** Task-1 XAI trên chính cohort và outer folds này đã được xem để hình thành giả thuyết. Tính train-only của code ngăn leakage trực tiếp, nhưng không xóa adaptive bias do người nghiên cứu đã xem kết quả trước đó. Kết quả này là exploratory reuse; cần external holdout để kiểm chứng khẳng định mới.

## Nuisance và thông tin có thể mất

Motor normalization là per-record không dùng nhãn: trừ điểm đầu ở x/y, chia chung theo RMS bán kính quỹ đạo quanh điểm đầu; pressure chia percentile95 của trị tuyệt đối. Guard tránh chia cho0. Giữ nguyên thời gian thực, pen state, tilt, azimuth. Nó hướng tới thay đổi vị trí, kích thước và gain pressure, nhưng cũng có thể mất kích thước chữ, biên độ vận động và mức lực liên quan impairment. Không gọi mọi biến thiên này là nhiễu vô hại.

| Biến đổi raw | Train augmentation / held-out stress | Thông tin có thể mất hoặc làm thay đổi |
|---|---|---|
| Translation | ±5% / ±10% bounding-box span ở hai trục | Vị trí/offset tuyệt đối; không đồng nghĩa đổi nội dung |
| Spatial scale | ×1,05 / ×1,10 quanh điểm đầu | Kích thước, vận tốc/acceleration tuyệt đối |
| Pressure gain | ×1,05 / ×1,10 | Mức lực có thể mang tín hiệu; chưa mô phỏng clipping/ADC |
| Sampling half | Giữ cách một sample và điểm cuối | Mất cao tần; không anti-alias; không tương đương mọi thiết bị |
| Timing warp | sin với biên độ1% /2% duration | Nhịp cục bộ thay đổi, monotonic derivative vẫn dương |
| Duration scale | ×1,05 / ×1,10 time từ thời điểm đầu | Tốc độ, thời lượng vốn có thể quan trọng cho nhãn |

Mỗi biến đổi gọi lại extractor gốc cho cả 14 tasks, kiểm tra finite và tính hợp lệ thời gian. Các stress được cố định trước phân tích và không dùng chọn cấu hình. Riêng `sampling_half` có cùng cường độ Train/Test; đây không phải thử ngoại suy toàn bộ nuisance. Train-augmented chọn bank ngẫu nhiên cho từng participant với xác suất0,3, chỉ index current Train; validation/test không augment. Các biến đổi là sensitivity hypotheses, **chưa chứng minh label-preserving**. Cache toàn cohort chỉ thực hiện biến đổi xác định không nhãn; mọi imputer/scaler/ranker/model vẫn fit Train riêng.

## Ablation kiểm soát và so sánh paired

| comparison | model | reference | oof_delta_f1_macro | mean_delta_f1_macro | std_delta_f1_macro | oof_delta_accuracy | mean_delta_accuracy | std_delta_accuracy | oof_delta_auc | mean_delta_auc | std_delta_auc | oof_delta_balanced_accuracy | mean_delta_balanced_accuracy | std_delta_balanced_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| self_attention | matched_transformer_no_type | matched_task_mlp | -0.0043 | -0.0042 | 0.0621 | -0.0039 | -0.0039 | 0.0578 | -0.0114 | -0.0054 | 0.0404 | -0.0043 | -0.0035 | 0.0608 |
| task_type_embedding | matched_transformer_type | matched_transformer_no_type | -0.0522 | -0.0569 | 0.0877 | -0.0504 | -0.0504 | 0.0875 | -0.0372 | -0.0479 | 0.0744 | -0.0514 | -0.0506 | 0.0860 |
| motor_normalization | matched_motor_normalized | matched_transformer_no_type | -0.0463 | -0.0509 | 0.0800 | -0.0465 | -0.0465 | 0.0721 | -0.0391 | -0.0297 | 0.0620 | -0.0454 | -0.0463 | 0.0775 |
| training_augmentation | matched_train_augmented | matched_transformer_no_type | 0.0376 | 0.0365 | 0.0391 | 0.0349 | 0.0349 | 0.0353 | 0.0206 | 0.0204 | 0.0288 | 0.0383 | 0.0379 | 0.0362 |
| train_only_task_selection | matched_xai_selected | matched_transformer_type | -0.0072 | -0.0048 | 0.0688 | -0.0078 | -0.0078 | 0.0685 | -0.0289 | -0.0397 | 0.0293 | -0.0070 | -0.0074 | 0.0641 |

Các matched controls không tham gia chọn nested winner. Cùng d/head/layer/pooling khi so sánh attention/normalization/augmentation/selection; epoch vẫn do inner validation của từng phương án. Task-MLP và Transformer có số tham số khác nhau, vì vậy so sánh attention chưa phải can thiệp nhân quả chỉ đổi duy nhất một cơ chế/capacity.

| model | corrected | harmed | unchanged_correct | unchanged_wrong | mean_delta_f1_macro | std_delta_f1_macro |
|---|---|---|---|---|---|---|
| lightgbm | 0 | 0 | 219 | 39 | 0.0000 | 0.0000 |
| lightgbm_without_text | 12 | 51 | 168 | 27 | -0.1553 | 0.0423 |
| motor_normalized | 23 | 54 | 165 | 16 | -0.1299 | 0.0803 |
| task_mlp | 23 | 48 | 171 | 16 | -0.1030 | 0.0539 |
| train_augmented | 24 | 51 | 168 | 15 | -0.1097 | 0.0505 |
| transformer | 23 | 50 | 169 | 16 | -0.1112 | 0.0100 |
| without_text_tasks | 21 | 57 | 162 | 18 | -0.1465 | 0.0581 |
| xai_selected | 18 | 38 | 181 | 21 | -0.0843 | 0.0541 |
| nested_selected | 22 | 54 | 165 | 17 | -0.1321 | 0.0210 |

Corrected = baseline sai, mô hình mới đúng; harmed = baseline đúng, mô hình mới sai. Cùng 258 participants, threshold probability >0,5 giữ nguyên, không chọn ngưỡng lại từ Test. Chênh lệch theo fold là paired; không dùng các fold phụ thuộc như những cohort độc lập để khẳng định significance.

## Overfitting, seeds và chi phí

| family | best_epoch | loss_gap | train_accuracy | validation_accuracy |
|---|---|---|---|---|
| motor_normalized | 15.3889 | 0.2021 | 0.8456 | 0.7108 |
| task_mlp | 16.6944 | 0.1760 | 0.8259 | 0.6963 |
| train_augmented | 16.0139 | 0.2034 | 0.8407 | 0.7028 |
| transformer | 15.3194 | 0.1970 | 0.8305 | 0.7100 |
| without_text_tasks | 9.6806 | 0.1260 | 0.7527 | 0.6624 |
| xai_selected | 14.6528 | 0.1895 | 0.8338 | 0.7045 |

Loss gap là validation BCE trừ Train BCE tại inner best epoch; Train loss/accuracy được đánh giá lại trong chế độ eval trên clean Train, không có dropout/task masking; loss gap vẫn chỉ là chẩn đoán trên tập Train/inner-validation nhỏ. Learning curves là mô tả, các epoch muộn có ít runs do early stopping. Outer model chỉ có Train curve; không có outer-Test loss curve dùng điều khiển fit.

| model | f1_macro_mean | f1_macro_std | accuracy_mean | accuracy_std | auc_mean | auc_std | balanced_accuracy_mean | balanced_accuracy_std |
|---|---|---|---|---|---|---|---|---|
| lightgbm | 0.8472 | 0.0000 | 0.8488 | 0.0000 | 0.9069 | 0.0000 | 0.8475 | 0.0000 |
| lightgbm_without_text | 0.6930 | 0.0000 | 0.6977 | 0.0000 | 0.7697 | 0.0000 | 0.6924 | 0.0000 |
| motor_normalized | 0.7141 | 0.0119 | 0.7183 | 0.0118 | 0.7959 | 0.0028 | 0.7136 | 0.0119 |
| nested_selected | 0.7171 | 0.0137 | 0.7209 | 0.0140 | 0.7821 | 0.0147 | 0.7168 | 0.0134 |
| task_mlp | 0.7411 | 0.0097 | 0.7468 | 0.0090 | 0.8211 | 0.0156 | 0.7395 | 0.0099 |
| train_augmented | 0.7414 | 0.0161 | 0.7442 | 0.0155 | 0.8095 | 0.0152 | 0.7417 | 0.0165 |
| transformer | 0.7393 | 0.0085 | 0.7442 | 0.0078 | 0.8117 | 0.0060 | 0.7380 | 0.0088 |
| without_text_tasks | 0.6905 | 0.0098 | 0.6977 | 0.0103 | 0.7656 | 0.0063 | 0.6892 | 0.0096 |
| xai_selected | 0.7526 | 0.0211 | 0.7558 | 0.0205 | 0.8301 | 0.0108 | 0.7525 | 0.0213 |

| model | parameters_min | parameters_max | train_seconds_mean | task_count_mean | inference_ms_per_person | timing_only_train_seconds |
|---|---|---|---|---|---|---|
| lightgbm | nan | nan | nan | 14.0000 | 0.0387 | 0.4055 |
| lightgbm_without_text | nan | nan | nan | 12.0000 | 0.0425 | nan |
| motor_normalized | 11041.0000 | 20066.0000 | 1.2546 | 14.0000 | 0.1124 | nan |
| task_mlp | 6817.0000 | 7265.0000 | 0.7019 | 14.0000 | 0.0820 | nan |
| train_augmented | 11041.0000 | 39361.0000 | 1.1441 | 14.0000 | 0.0956 | nan |
| transformer | 11041.0000 | 39361.0000 | 0.6603 | 14.0000 | 0.1116 | nan |
| without_text_tasks | 11041.0000 | 39361.0000 | 0.6686 | 12.0000 | 0.0894 | nan |
| xai_selected | 20066.0000 | 39361.0000 | 1.3859 | 8.5000 | 0.1148 | nan |
| nested_selected | nan | nan | nan | nan | nan | nan |

Neural parameter count và tree complexity không đồng nhất: không gán cây một số neural parameters giả. Train cost neural đo vòng tối ưu của mỗi final fit, không gồm fit preprocessor/khởi tạo và inference là median20 warm measurements cho batch43, gồm cached features→preprocessor→GPU transfer→probability. LightGBM chạy CPU, neural trên Quadro T1000; không diễn giải đây là so sánh tốc độ công bằng phần cứng. Raw preparation cho toàn bộ 19 banks mất 718.2s với ba CPU workers, không nằm trong inference latency. Pilot kiểm tra fit GPU với batch64; chi phí pipeline thực gồm inner search, ranker, cache và attribution, không chỉ một forward pass.

## Robustness khi nuisance và thiếu tác vụ

| model | condition | delta_f1_macro_mean | delta_f1_macro_std | mean_absolute_probability_shift_mean | attribution_rank_spearman_mean |
|---|---|---|---|---|---|
| lightgbm | duration_scale | -0.0135 | 0.0418 | 0.0505 | 0.9912 |
| lightgbm | pressure_scale | -0.0002 | 0.0004 | 0.0103 | 0.9971 |
| lightgbm | sampling_half | -0.2490 | 0.1101 | 0.2483 | 0.9553 |
| lightgbm | spatial_scale | -0.0784 | 0.0431 | 0.1433 | 0.9905 |
| lightgbm | timing_warp | -0.0267 | 0.0351 | 0.0574 | 0.9919 |
| lightgbm | translation | -0.0210 | 0.0235 | 0.0274 | 0.9963 |
| motor_normalized | duration_scale | -0.0166 | 0.0307 | 0.0354 | 0.9451 |
| motor_normalized | pressure_scale | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| motor_normalized | sampling_half | -0.0806 | 0.1125 | 0.1981 | 0.7209 |
| motor_normalized | spatial_scale | 0.0000 | 0.0000 | 0.0001 | 1.0000 |
| motor_normalized | timing_warp | 0.0036 | 0.0493 | 0.0833 | 0.8989 |
| motor_normalized | translation | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| train_augmented | duration_scale | -0.0223 | 0.0254 | 0.0366 | nan |
| train_augmented | pressure_scale | -0.0116 | 0.0129 | 0.0323 | nan |
| train_augmented | sampling_half | -0.0922 | 0.0302 | 0.1583 | nan |
| train_augmented | spatial_scale | -0.0023 | 0.0226 | 0.0345 | nan |
| train_augmented | timing_warp | -0.0361 | 0.0608 | 0.0874 | nan |
| train_augmented | translation | -0.0023 | 0.0442 | 0.0640 | nan |
| transformer | duration_scale | -0.0145 | 0.0317 | 0.0460 | 0.9048 |
| transformer | pressure_scale | 0.0090 | 0.0394 | 0.0358 | 0.9744 |
| transformer | sampling_half | -0.2527 | 0.0973 | 0.2631 | 0.4418 |
| transformer | spatial_scale | -0.0031 | 0.0350 | 0.0394 | 0.9678 |
| transformer | timing_warp | -0.0097 | 0.0303 | 0.0861 | 0.8476 |
| transformer | translation | -0.0228 | 0.0409 | 0.0704 | 0.9634 |

Delta là stress trừ clean trên cùng held-out fold. Probability shift nhỏ nhưng clean performance thấp không chứng minh mô hình tốt hơn. Attribution rank stress chỉ tính cho LightGBM, raw Transformer và motor-normalized Transformer; ô trống ở family khác không phải correlation0. Kiểm tra riêng spatial invariance của phép chuẩn hóa ở raw arrays trong unit tests không thay thế kết quả dự đoán thực.

| model | kind | drop_f1_macro_mean | drop_f1_macro_std |
|---|---|---|---|
| lightgbm | missing_2 | 0.0378 | 0.0175 |
| lightgbm | missing_4 | 0.0559 | 0.0177 |
| lightgbm | missing_6 | 0.1079 | 0.0389 |
| lightgbm | one_task | 0.0207 | 0.0131 |
| lightgbm | without_text_diagnostic | 0.1750 | 0.0696 |
| lightgbm_without_text | missing_2 | 0.0043 | 0.0364 |
| lightgbm_without_text | missing_4 | 0.0086 | 0.0490 |
| lightgbm_without_text | missing_6 | 0.0348 | 0.0426 |
| lightgbm_without_text | one_task | 0.0011 | 0.0233 |
| lightgbm_without_text | without_text_diagnostic | 0.0000 | 0.0000 |
| motor_normalized | missing_2 | 0.0301 | 0.0340 |
| motor_normalized | missing_4 | 0.0256 | 0.0509 |
| motor_normalized | missing_6 | 0.0599 | 0.0435 |
| motor_normalized | one_task | 0.0057 | 0.0201 |
| motor_normalized | without_text_diagnostic | 0.1174 | 0.0880 |
| task_mlp | missing_2 | 0.0165 | 0.0245 |
| task_mlp | missing_4 | 0.0178 | 0.0388 |
| task_mlp | missing_6 | 0.0466 | 0.0508 |
| task_mlp | one_task | 0.0022 | 0.0263 |
| task_mlp | without_text_diagnostic | 0.0240 | 0.0363 |
| train_augmented | missing_2 | 0.0281 | 0.0172 |
| train_augmented | missing_4 | 0.0376 | 0.0108 |
| train_augmented | missing_6 | 0.0640 | 0.0248 |
| train_augmented | one_task | 0.0155 | 0.0100 |
| train_augmented | without_text_diagnostic | 0.0645 | 0.0396 |
| transformer | missing_2 | 0.0093 | 0.0304 |
| transformer | missing_4 | 0.0138 | 0.0303 |
| transformer | missing_6 | 0.0318 | 0.0187 |
| transformer | one_task | -0.0025 | 0.0204 |
| transformer | without_text_diagnostic | 0.0268 | 0.0576 |
| without_text_tasks | missing_2 | 0.0027 | 0.0155 |
| without_text_tasks | missing_4 | 0.0126 | 0.0187 |
| without_text_tasks | missing_6 | 0.0223 | 0.0184 |
| without_text_tasks | one_task | 0.0063 | 0.0101 |
| without_text_tasks | without_text_diagnostic | 0.0000 | 0.0000 |
| xai_selected | missing_2 | 0.0292 | 0.0221 |
| xai_selected | missing_4 | 0.0408 | 0.0408 |
| xai_selected | missing_6 | 0.0749 | 0.0526 |
| xai_selected | one_task | 0.0111 | 0.0112 |
| xai_selected | without_text_diagnostic | 0.1670 | 0.0677 |

Đã thử14 single-task masks, copy+dictation mask, và 2/4/6 tasks với10 subset draws/fold; dùng cùng subset cho mọi mô hình. Nets dùng missing-token mask, trees thay bằng median Train: khác cơ chế nên so sánh có giới hạn. Tasks đã không được model chọn có thể bị mask mà không đổi prediction; không được suy thành độ bền với cùng số task đang hoạt động bị mất. Kiểm tra độc lập zero-feature ablation giữ token present và đặt Z=0 (Train mean sau scaling); trees dùng Train raw mean. Hai phép kiểm tra này được lưu riêng.

## XAI: logit IG, task/group và kiểm chứng độc lập

IG dùng positive-class **logit**, reference là vector chuẩn hóa0 tương ứng mean Train; giữ cùng task mask và task-ID embeddings. Tổng attribution khớp `logit(x)-logit(reference)` với tolerance0,002 +0,002×|delta logit| trên mỗi participant. Dùng Gauss–Legendre quadrature32→512 bước, tăng cho tới khi hội tụ, không đổi ngưỡng theo Test. 108 artifacts clean/stress, tối đa residual tuyệt đối 0.002776; số bước thực tế từ 32 đến 512. Attribution không phải xác suất và không tự quy đổi từng chiều qua sigmoid. TreeSHAP tái dùng exact saved baseline và kiểm tra raw-score additivity; references giữa TreeSHAP và IG khác nhau, so sánh rank/share, không coi absolute mass là cùng chuẩn đo sinh học.

Mapping task và12 feature groups lấy đúng Task-1 API, bao gồm velocity, acceleration, jerk, pressure, orientation, geometry/curvature, timing, pen-state và FFT. Một số khái niệm như smoothness được phản ánh gián tiếp qua jerk/geometry; không gán nhãn sinh học mới cho feature chưa có phép đo tương ứng. CSV group có `n_features` và mass/feature để thấy ảnh hưởng kích thước nhóm.

Top5 tasks theo từng mô hình:

| model | block | share_percent | rank | fold_sd |
|---|---|---|---|---|
| lightgbm | copy | 35.3414 | 1.0000 | 0.1203 |
| lightgbm | dictation | 11.2088 | 2.0000 | 0.2253 |
| lightgbm | lower-loops-fast | 8.3401 | 3.0000 | 0.2080 |
| lightgbm | small-spiral-fast | 7.3521 | 4.0000 | 0.1481 |
| lightgbm | RCFT-recall | 7.1436 | 5.0000 | 0.2420 |
| motor_normalized | copy | 13.4875 | 1.0000 | 1.9990 |
| motor_normalized | dictation | 8.6134 | 2.0000 | 0.7373 |
| motor_normalized | rainbow-fast | 8.1629 | 3.0000 | 0.7465 |
| motor_normalized | lower-loops-fast | 7.5940 | 4.0000 | 0.7102 |
| motor_normalized | RCFT-recognition | 7.0677 | 5.0000 | 0.6748 |
| transformer | copy | 7.7991 | 1.0000 | 1.0434 |
| transformer | combined-loops-precise | 7.6810 | 2.0000 | 0.6762 |
| transformer | RCFT-recognition | 7.4929 | 3.0000 | 0.7837 |
| transformer | dictation | 7.4659 | 4.0000 | 0.7252 |
| transformer | small-spiral-precise | 7.4640 | 5.0000 | 0.7596 |
| xai_selected | copy | 31.3933 | 1.0000 | 3.7588 |
| xai_selected | lower-loops-fast | 13.8042 | 2.0000 | 1.1782 |
| xai_selected | dictation | 13.1547 | 3.0000 | 1.2387 |
| xai_selected | RCFT-recognition | 8.5177 | 4.0000 | 0.8730 |
| xai_selected | lower-loops-precise | 7.4018 | 5.0000 | 1.5590 |

Top5 feature groups:

| model | block | n_features | share_percent | mass_per_feature |
|---|---|---|---|---|
| lightgbm | fft | 224 | 27.9621 | 0.0152 |
| lightgbm | statistics | 140 | 21.5927 | 0.0188 |
| lightgbm | trajectory | 112 | 20.2745 | 0.0220 |
| lightgbm | orientation | 56 | 8.5462 | 0.0186 |
| lightgbm | pressure | 112 | 4.8435 | 0.0053 |
| motor_normalized | fft | 224 | 26.0385 | 0.0359 |
| motor_normalized | statistics | 140 | 14.4473 | 0.0319 |
| motor_normalized | pressure | 112 | 10.4339 | 0.0288 |
| motor_normalized | trajectory | 112 | 10.0436 | 0.0277 |
| motor_normalized | acceleration | 84 | 8.0767 | 0.0297 |
| transformer | fft | 224 | 25.1803 | 0.0457 |
| transformer | statistics | 140 | 16.7154 | 0.0486 |
| transformer | acceleration | 84 | 10.0407 | 0.0486 |
| transformer | trajectory | 112 | 9.8846 | 0.0359 |
| transformer | pressure | 112 | 7.9376 | 0.0288 |

Độ ổn định giữa15 cặp outer folds (mean Spearman/Top3 Jaccard; các cặp phụ thuộc):

| model | level | spearman | top3_jaccard |
|---|---|---|---|
| lightgbm | group | 0.9133 | 1.0000 |
| lightgbm | task | 0.8221 | 0.5667 |
| motor_normalized | group | 0.9096 | 0.6333 |
| motor_normalized | task | 0.3037 | 0.3133 |
| task_mlp | group | 0.9417 | 1.0000 |
| task_mlp | task | 0.3067 | 0.3600 |
| train_augmented | group | 0.8718 | 0.8333 |
| train_augmented | task | 0.2176 | 0.3000 |
| transformer | group | 0.9352 | 0.7000 |
| transformer | task | 0.3623 | 0.3000 |
| without_text_tasks | group | 0.8914 | 0.6333 |
| without_text_tasks | task | 0.4863 | 0.3200 |
| xai_selected | group | 0.8172 | 1.0000 |
| xai_selected | task | 0.6983 | 0.6800 |

Faithfulness: rank attribution so với macro-F1 drop khi grouped permutation (10 repeats/task/group/fold) và zero-feature ablation. Có12.480 held-out permutation evaluations, dùng chung hoán vị giữa mô hình. Negative importance/drop được giữ nguyên. Perturbation có thể ra ngoài distribution hoặc chịu feature redundancy; correlation thấp phải được báo cáo, không sửa ranking để phù hợp kỳ vọng.

| model | level | comparison | spearman |
|---|---|---|---|
| lightgbm | task | attribution_vs_permutation | 0.9209 |
| lightgbm | task | attribution_vs_zero_feature_ablation | 0.7802 |
| lightgbm | group | attribution_vs_permutation | 0.9091 |
| task_mlp | task | attribution_vs_permutation | -0.2176 |
| task_mlp | task | attribution_vs_zero_feature_ablation | 0.1560 |
| task_mlp | group | attribution_vs_permutation | 0.8601 |
| transformer | task | attribution_vs_permutation | 0.6352 |
| transformer | task | attribution_vs_zero_feature_ablation | 0.6044 |
| transformer | group | attribution_vs_permutation | 0.7902 |
| motor_normalized | task | attribution_vs_permutation | 0.7758 |
| motor_normalized | task | attribution_vs_zero_feature_ablation | 0.6264 |
| motor_normalized | group | attribution_vs_permutation | 0.2308 |
| xai_selected | task | attribution_vs_permutation | 0.8813 |
| xai_selected | task | attribution_vs_zero_feature_ablation | 0.7925 |
| xai_selected | group | attribution_vs_permutation | 0.6643 |
| train_augmented | task | attribution_vs_permutation | 0.4945 |
| train_augmented | task | attribution_vs_zero_feature_ablation | 0.4769 |
| train_augmented | group | attribution_vs_permutation | 0.7343 |
| without_text_tasks | task | attribution_vs_permutation | 0.6960 |
| without_text_tasks | task | attribution_vs_zero_feature_ablation | 0.6696 |
| without_text_tasks | group | attribution_vs_permutation | 0.7273 |

Rank disagreement với LightGBM:

| model | level | reference | spearman |
|---|---|---|---|
| task_mlp | task | lightgbm_shap | -0.2967 |
| transformer | task | lightgbm_shap | 0.3670 |
| motor_normalized | task | lightgbm_shap | 0.4374 |
| xai_selected | task | lightgbm_shap | 0.8637 |
| train_augmented | task | lightgbm_shap | 0.4725 |
| without_text_tasks | task | lightgbm_shap | -0.3256 |
| task_mlp | group | lightgbm_shap | 0.8042 |
| transformer | group | lightgbm_shap | 0.6993 |
| motor_normalized | group | lightgbm_shap | 0.7063 |
| xai_selected | group | lightgbm_shap | 0.9091 |
| train_augmented | group | lightgbm_shap | 0.7413 |
| without_text_tasks | group | lightgbm_shap | 0.7832 |

Attention chỉ là chẩn đoán phụ, không tự coi là explanation. Incoming attention được trung bình qua layers/heads và valid queries; pooling weights lưu riêng, missing keys0. Attention-vs-IG:

| model | comparison | spearman |
|---|---|---|
| motor_normalized | incoming_attention_vs_IG | 0.6220 |
| train_augmented | incoming_attention_vs_IG | 0.6220 |
| transformer | incoming_attention_vs_IG | 0.9341 |
| without_text_tasks | incoming_attention_vs_IG | 0.9339 |
| xai_selected | incoming_attention_vs_IG | 0.9473 |

Các ca local trong `tables/local_cases.csv` chọn ID nhỏ nhất theo thứ tự từ điển cho từng outcome của raw Transformer, sau khi hoàn tất dự đoán. Đây là minh họa hậu nghiệm, không dùng để đổi model. Có bảng signed/absolute task IG cho toàn bộ participants để kiểm tra cả đúng, sai, corrected và harmed; không chỉ trình bày ca đẹp.

## Trả lời câu hỏi nghiên cứu

**RQ1 — Self-attention có cải thiện tương tác task?** Matched attention so với Task-MLP: Δmacro-F1 OOF -0.0043; paired fold mean -0.0042 ± 0.0621. Đây là bằng chứng giới hạn trong grid và cohort này; không thấy lợi ích dự đoán từ matched attention.

**RQ2 — Motor processing có bền mà không mất thông tin?** Matched normalization Δmacro-F1 OOF -0.0463, augmentation +0.0376. Cần đọc đồng thời clean metrics và nuisance table: phép normalize có thể triệt scale/gain theo thiết kế nhưng lấy đi thông tin dự đoán. Không gọi nó là cải thiện miễn phí; không thay baseline chỉ từ invariance toán học.

**RQ3 — Giải thích ổn định/faithful hơn?** Xem riêng hai bảng stability và faithfulness ở trên; không gộp thành một tiêu chí thuận lợi duy nhất. IG completeness chỉ chứng minh phép tích phân khớp đầu ra, không chứng minh clinical validity. Stability cao hơn nếu có cũng không bù được dự đoán thấp hơn; attention khác IG là kết quả cần phân tích, không phải lý do ép rank khớp.

**RQ4 — Ít tasks có cạnh tranh?** XAI-selected dùng trung bình 8.50 tasks, macro-F1 0.7680 (Δbaseline -0.0792); without copy/dictation dùng12 tasks đạt 0.7018 (Δbaseline -0.1454). Kết quả tree without-text được báo cáo song song. Không kết luận số tasks tối thiểu từ SHAP/IG hoặc một vài k; cần task-ablation protocol riêng trên dữ liệu độc lập. Bỏ copy/dictation chỉ kiểm tra phụ thuộc hai task nhiều nội dung chữ, không phải kiểm định language invariance.

**RQ5 — Generalization dưới nuisance?** Chỉ chứng minh mức nhạy thực nghiệm với sáu biến đổi cố định đã nêu, không phải generalization tới thiết bị, bệnh viện, ngôn ngữ hoặc cohort mới. Timing/sampling có thể tác động ngược nhãn; phương án robust nhưng accuracy thấp không tự là mô hình phù hợp hơn. Không chọn lại nuisance magnitudes/thresholds/model từ các outer stress results.


### Kiểm tra số học đặc trưng cực đoan

Trong outer fold4, participant NB05301 có một tọa độ sau Train standardization khoảng 2,64×10^13. Giữ nguyên extractor, scaler, checkpoint và prediction; không clip lại sau khi xem Test. Global Gauss–Legendre512 bỏ lỡ chuyển tiếp logit rất sát alpha0, tạo residual IG khoảng1,629 ở Task-MLP. Đã bổ sung quadrature trên geometric alpha panels cho đúng các participant chưa hội tụ; không nới tolerance và không dùng nhãn để chọn phương pháp tích phân. Có unit test tanh với input10^13 kiểm tra độc lập. Chênh lệch input cực đoan này cũng cho thấy scaling và neural generalization có giới hạn; nên nghiên cứu preprocessing robust bằng protocol mới trong tương lai, không sửa mô hình của thực nghiệm đã khóa.

Các artifact phải tinh chỉnh tích phân (CSV/NPZ còn lưu global residual ban đầu, indices, panel order/count):

| artifact | participants | panel_order |
|---|---|---|
| analysis/fold_4/motor_normalized_ig.npz | 1 | 32 |
| analysis/fold_4/motor_normalized_ig_duration_scale.npz | 1 | 32 |
| analysis/fold_4/motor_normalized_ig_pressure_scale.npz | 1 | 32 |
| analysis/fold_4/motor_normalized_ig_sampling_half.npz | 1 | 32 |
| analysis/fold_4/motor_normalized_ig_spatial_scale.npz | 1 | 32 |
| analysis/fold_4/motor_normalized_ig_timing_warp.npz | 1 | 32 |
| analysis/fold_4/motor_normalized_ig_translation.npz | 1 | 32 |
| analysis/fold_4/task_mlp_ig.npz | 1 | 32 |
| analysis/fold_4/train_augmented_ig.npz | 1 | 32 |
| analysis/fold_4/transformer_ig.npz | 1 | 32 |
| analysis/fold_4/transformer_ig_duration_scale.npz | 1 | 32 |
| analysis/fold_4/transformer_ig_pressure_scale.npz | 1 | 32 |
| analysis/fold_4/transformer_ig_sampling_half.npz | 1 | 32 |
| analysis/fold_4/transformer_ig_spatial_scale.npz | 1 | 32 |
| analysis/fold_4/transformer_ig_timing_warp.npz | 1 | 32 |
| analysis/fold_4/transformer_ig_translation.npz | 1 | 32 |
| analysis/fold_4/without_text_tasks_ig.npz | 1 | 32 |

### Diễn giải định lượng bổ sung

Raw Transformer có mean task-rank stability rho=0.3623, thấp hơn LightGBM 0.8221; agreement attribution–permutation cũng thấp hơn (0.6352 so với 0.9209). Chưa có bằng chứng Transformer giải thích ổn định hoặc faithful hơn baseline. XAI-selected có Top3 Jaccard cao hơn (0.6800), nhưng zero-attribution ở tasks bị loại và lựa chọn task từ LightGBM có thể làm agreement/stability cao theo thiết kế; không coi đó là cải thiện giải thích độc lập hoặc lý do bỏ qua accuracy thấp hơn.

Motor-normalized giữ macro-F1 không đổi trước translation/spatial-scale/pressure-gain đã thử, trong khi raw Transformer vẫn nhạy với một số biến đổi; probability shifts không hoàn toàn0 do floating point và extractor. Đổi lại clean macro-F1 của motor-normalized (0.7239) thấp hơn raw Transformer (0.7385) và baseline (0.8472); matched normalization cũng giảm điểm. Sampling half là stress mạnh: baseline giảm mean fold F1 khoảng0,2490, raw Transformer0,2527, motor-normalized0,0806, train-augmented0,0922. Đây là độ giảm tương đối; đọc cả mức điểm tuyệt đối, không kết luận tổng quát rằng neural tốt hơn.

XAI-selected có OOF F1 cao hơn raw Transformer trong bảng family, nhưng matched top10 so với full14 cùng cấu hình giảm F1; chưa thể gán lợi ích family cho riêng task selection. Task-ID embedding cũng giảm F1 trong matched comparison. Matched augmentation có tín hiệu tích cực nhưng không được chọn làm “winner đã định trước” bằng cách nhìn outer Test; giữ nó là giả thuyết cho kiểm chứng độc lập.

## Hình và bảng nên giữ cho paper

Giữ architecture (01), predictive ablation (02), matched ablation (03), seeds (04), curves (05), compute (06), nuisance (07), missing tasks (08), task/group attribution (09), stability (10), faithfulness (11), confusion (13), corrected/harmed (14). Hình12 attention đặt supplementary và ghi rõ không phải faithful attribution. Mọi hình có PNG300dpi/SVG, caption/nguồn/hashes trong `figures/manifest.json`. Đây là paper drafts; phần diễn giải phải giữ negative results và giới hạn exploratory cohort reuse.

Notebook `notebooks/06_kinematic_task_transformer.ipynb` ưu tiên các **cell bảng số kết quả cuối cùng**, sau đó mới hiển thị hình. Các CSV chính: `ablation_primary`, `matched_ablation`, `fold_results`, `oof_predictions`, `paired_comparisons`, `robustness_summary`, `masking_summary`, `attribution_importance`, `attribution_stability`, `faithfulness`, `seed_sensitivity_summary`, `inference_cost`. Không có chỉ số do ảnh tạo sinh hoặc ước lượng từ biểu đồ.

## Tái lập và validation

Chạy bằng môi trường hiện có, từ project root:

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/prepare_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/train_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/train_task_transformer_controls.py
.venv/Scripts/python.exe -X utf8 -B scripts/analyze_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/report_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/write_task_transformer_report.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_task_transformer_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/validate_task_transformer.py --numerics-only
.venv/Scripts/python.exe -X utf8 -B scripts/plot_task_transformer.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 06_kinematic_task_transformer.ipynb --validation-output results/task_transformer/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_task_transformer.py
```

Raw data/cache/checkpoints không đưa lên Git; checkpoint Task2 ở `results/task_transformer/checkpoints/`, được giữ local, hash được lưu trong JSON. Các bước fit resumable theo training signature; không sửa config/code huấn luyện rồi âm thầm dùng checkpoint cũ. `numeric_validation.json` tái chạy inference từ mọi checkpoint, kiểm tra Train medians/scaler, IDs, choices/epochs inner, baseline OOF, nuisance/masks/zero-feature/permutation probabilities và IG logits/completeness; đóng băng hash artifacts và code. Final validation còn kiểm tra chín unit tests, notebook execution, hình/nguồn và lịch sử không đổi.

Đề xuất thực tế: giữ baseline LightGBM làm mốc chính; giữ pipeline Transformer nhẹ và các kết quả âm/dương như nghiên cứu ablation. Chỉ xem xét thay đổi hướng mô hình tiếp theo sau review, không triển khai thêm kiến trúc, Fusion, Knowledge Graph hay cross-language trong nhiệm vụ này. Giải thích mô hình không phải kết luận nhân quả hoặc chẩn đoán lâm sàng.
