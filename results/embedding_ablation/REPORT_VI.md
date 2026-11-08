# Embedding Extraction và Ablation Study — DiaGraMo

## Phạm vi và trạng thái

Đánh giá GAF/ViT-B16 và CWT/EfficientNet-B4; không thực hiện Fusion. Giữ nguyên **258 participant,
nhãn group, Stratified 6-Fold, shuffle=True, seed 42**, cùng fold assignments cũ.
Trạng thái hoàn thành cuối cùng chỉ xác nhận bởi `validation.json`: số liệu, notebook, hình và test đều phải qua.

## Vấn đề của baseline và ba phương án

Kernel gốc `W7[:,c] = sum_RGB(W3)/7` giống nhau ở cả bảy kênh. Với chuẩn hóa chung
`(I−0.449)/0.226`, convolution gốc tương đương encoder RGB nhận trung bình bảy ảnh kênh
lặp lại ở R/G/B. Nó bất biến với hoán vị bảy kênh, nên không giữ danh tính kênh ở lớp đầu.
Unit test kiểm tra đại số; trích xuất kiểm tra số trên hai tác vụ đầu với checkpoint thật và embedding cũ.
`baseline_extraction_audit.json` còn xác nhận tái trích xuất đủ **3.612 tác vụ mỗi nhánh** khớp embedding gốc
trong tolerance số học (atol=rtol=3e-4), không ghi đè ma trận gốc. Điều này kiểm tra sự tương thích
của ảnh/cache, thứ tự hàng và checkpoint cho toàn cohort, ngoài kiểm chứng đại số trên hai tác vụ.

- `original`: embedding gốc, dự đoán OOF/inner scores được giữ và đọc lại nguyên vẹn.
- `group_adapter`: 1×1 convolution cố định 7→3; R=mean(x,y), G=mean(pressure,pen_status),
  B=mean(tilt,azimuth,time). Mỗi input tham gia; các hàng là convex average. Phép gộp này vẫn mất
  thông tin và việc gán nhóm vào RGB là giả định cố định, chưa phải ánh xạ tối ưu đã học.
- `channel_mean`: encode riêng từng ảnh kênh bằng encoder RGB không sửa kernel, rồi trung bình bảy
  embedding sau nonlinear encoder. Số chiều bằng baseline, chi phí bảy lượt encoder.
- `channel_concat`: cùng bảy embedding trên nhưng nối theo thứ tự x/y/pressure/tilt/azimuth/time/pen_status.
  Giữ danh tính kênh; số chiều gấp bảy. Thứ tự vẫn là participant → task → channel → embedding coordinate.

Tất cả encoder **đóng băng, eval(), float32**; adapter không có tham số trainable và không đọc nhãn.
Không có adapter học nhãn trên toàn cohort. Cache chỉ chứa embedding label-free; không fine-tune encoder.
Giữ nguyên ảnh 224×224, định nghĩa GAF/CWT và scalar normalization cũ để cô lập channel adaptation.
Đây không phải thử nghiệm chuẩn hóa ImageNet RGB hay tối ưu resolution mới.

## Protocol ablation và chống leakage

Lưới chung: **192 cấu hình** = 2 nhánh × 4 embedding × 8 reducer × 3 classifier.
Reducer: `none`, PCA-16/32, PCA-95%/99%, ANOVA-64/256 và ExtraTrees-256.
Classifier: Logistic Regression, SVM, LightGBM, nguyên siêu tham số baseline.
PCA/selector dùng `ReductionBank` của nghiên cứu trước: imputer → loại cột hằng → StandardScaler → reducer,
tất cả fit chỉ trên train của split tương ứng. PCA không whitening hoặc scale lại sau projection.

Mỗi outer fold có inner Stratified 3-Fold mới trên 215 người train; toàn bộ transform được fit lại
trong inner train. Hàm chọn chỉ nhận inner scores, xếp theo F1 macro rồi Accuracy, AUC, ít chiều hơn và tên.
Không chọn threshold từ test. `original_raw` dùng inner CV chọn classifier; `original_compact` giữ toàn bộ
54 ứng viên cũ (18 reducer kể cả none × 3 classifier), dự đoán nested khớp chính xác đợt trước.
Mỗi embedding mới có 24 ứng viên; `new_selected` chọn giữa 72 ứng viên mới;
`all_selected` chọn giữa 126 ứng viên mới+cũ, cho phép baseline thắng. Pool chọn có kích thước khác nhau,
nên không diễn giải chênh lệch như tác động thuần của số lượng ứng viên.

## Bảng ablation cấu hình cố định — thăm dò

Mỗi hàng chọn classifier/reducer theo chính outer grid, **có selection bias**. Nhóm “có” loại `none`.
Đơn vị %, bộ ba **mean Accuracy / mean F1 macro / mean ROC-AUC của 6 fold**.
SD và pooled OOF được giữ trong `summary.csv`; không trộn hai cách tổng hợp.

| Nhánh | Embedding | Giảm_chiều | Cấu_hình | Chiều | Acc_F1_AUC |
| --- | --- | --- | --- | --- | --- |
| GAF | original | không | none / LightGBM | 10752.00 | 63.18 / 61.65 / 66.30 |
| GAF | original | có | anova_256 / SVM | 256.00 | 67.05 / 66.00 / 66.17 |
| GAF | group_adapter | không | none / LightGBM | 10752.00 | 69.38 / 68.20 / 71.64 |
| GAF | group_adapter | có | anova_256 / LightGBM | 256.00 | 70.54 / 69.85 / 73.60 |
| GAF | channel_mean | không | none / LightGBM | 10752.00 | 68.60 / 67.45 / 72.69 |
| GAF | channel_mean | có | anova_64 / SVM | 64.00 | 70.93 / 69.77 / 77.79 |
| GAF | channel_concat | không | none / LightGBM | 75264.00 | 74.42 / 73.39 / 80.87 |
| GAF | channel_concat | có | anova_256 / LightGBM | 256.00 | 76.36 / 75.65 / 82.37 |
| CWT | original | không | none / LogisticRegression | 25088.00 | 68.60 / 68.10 / 75.85 |
| CWT | original | có | pca_var99 / LogisticRegression | 203.33 | 71.32 / 70.74 / 76.07 |
| CWT | group_adapter | không | none / LogisticRegression | 25088.00 | 66.28 / 65.98 / 70.49 |
| CWT | group_adapter | có | pca_var95 / SVM | 168.33 | 67.05 / 66.04 / 70.95 |
| CWT | channel_mean | không | none / LogisticRegression | 25088.00 | 68.99 / 68.76 / 74.78 |
| CWT | channel_mean | có | pca_var99 / LogisticRegression | 203.00 | 70.16 / 69.72 / 73.21 |
| CWT | channel_concat | không | none / LogisticRegression | 175616.00 | 70.93 / 70.78 / 78.22 |
| CWT | channel_concat | có | anova_256 / SVM | 256.00 | 73.64 / 72.68 / 79.58 |

Tác động riêng tại **cùng classifier và cùng reducer**: `ablation_matched.csv` có
`delta_adaptation_*_pp` so với embedding gốc cùng reducer; `delta_reduction_*_pp` so với
cùng embedding, giữ nguyên chiều. Đây là đối chiếu đủ ô, không chỉ đối chiếu các hàng thắng khác classifier.
`channel_mean` so với `channel_concat` cô lập giữ danh tính kênh sau encoder với cùng số lượt inference;
số chiều cũng thay đổi nên cần xem thêm các reducer cùng ngân sách.

## Nested CV — ưu tiên để đánh giá lợi ích thực tế

Mọi lựa chọn bên trong train. `raw` chỉ chọn classifier; `compact` cho phép chọn `none`.
Các delta là điểm phần trăm F1 so với nested gốc và nested giảm chiều cũ, cùng outer participant.

| Nhánh | Quy_trình | Chiều_TB | Acc_F1_AUC | Δ_F1_vs_gốc | Δ_F1_vs_giảm_chiều_cũ |
| --- | --- | --- | --- | --- | --- |
| GAF | original_raw | 10752.00 | 62.40 / 60.95 / 66.27 | 0.00 | 5.67 |
| GAF | original_compact | 133.33 | 56.59 / 55.28 / 61.05 | -5.67 | 0.00 |
| GAF | group_adapter_raw | 10752.00 | 63.57 / 62.62 / 68.90 | 1.67 | 7.34 |
| GAF | group_adapter_compact | 195.00 | 64.34 / 63.65 / 74.41 | 2.70 | 8.37 |
| GAF | channel_mean_raw | 10752.00 | 65.89 / 64.28 / 72.46 | 3.33 | 9.00 |
| GAF | channel_mean_compact | 1863.50 | 72.09 / 71.18 / 76.54 | 10.23 | 15.90 |
| GAF | channel_concat_raw | 75264.00 | 73.26 / 72.65 / 80.94 | 11.70 | 17.37 |
| GAF | channel_concat_compact | 233.83 | 73.26 / 72.70 / 79.61 | 11.75 | 17.42 |
| GAF | new_selected | 244.83 | 73.26 / 72.37 / 79.39 | 11.42 | 17.10 |
| GAF | all_selected | 244.83 | 73.26 / 72.37 / 79.39 | 11.42 | 17.10 |
| CWT | original_raw | 25088.00 | 68.60 / 68.10 / 75.85 | 0.00 | 3.31 |
| CWT | original_compact | 216.67 | 65.89 / 64.79 / 70.45 | -3.31 | 0.00 |
| CWT | group_adapter_raw | 25088.00 | 66.28 / 65.98 / 70.49 | -2.12 | 1.19 |
| CWT | group_adapter_compact | 12663.17 | 66.67 / 66.38 / 69.18 | -1.73 | 1.59 |
| CWT | channel_mean_raw | 25088.00 | 66.28 / 65.51 / 72.82 | -2.60 | 0.72 |
| CWT | channel_mean_compact | 8486.67 | 61.24 / 60.38 / 66.03 | -7.73 | -4.41 |
| CWT | channel_concat_raw | 175616.00 | 66.28 / 65.90 / 73.91 | -2.20 | 1.12 |
| CWT | channel_concat_compact | 208.00 | 65.89 / 64.98 / 71.75 | -3.13 | 0.19 |
| CWT | new_selected | 208.00 | 65.89 / 64.98 / 71.75 | -3.13 | 0.19 |
| CWT | all_selected | 208.00 | 64.34 / 63.05 / 67.45 | -5.06 | -1.74 |

## Chi phí tính toán

Benchmark chung: 32 tác vụ đầu, không dùng nhãn, batch 16, warmup một lần/mode, lặp ba lần,
báo median. Bao gồm transfer CPU↔GPU, encoder và aggregation; **không gồm** tạo ảnh, load model,
fit reducer hoặc classifier. ms/participant là ước tính tuyến tính từ throughput batch × 14 tác vụ,
không phải latency thực đo của một participant đơn lẻ. Bộ nhớ là peak allocated của PyTorch,
không bao gồm toàn bộ VRAM/driver; hai encoder RGB/7ch cùng được load cho benchmark.

| branch | mode | raw_features | median_seconds | estimated_ms_per_participant | peak_gpu_allocated_mb |
| --- | --- | --- | --- | --- | --- |
| GAF | original | 10752 | 0.84 | 367.51 | 829.76 |
| GAF | group_adapter | 10752 | 0.83 | 365.17 | 816.95 |
| GAF | channel_mean | 10752 | 5.79 | 2532.24 | 881.26 |
| GAF | channel_concat | 75264 | 6.06 | 2650.74 | 881.26 |
| CWT | original | 25088 | 0.19 | 82.27 | 427.64 |
| CWT | group_adapter | 25088 | 0.19 | 81.42 | 414.82 |
| CWT | channel_mean | 25088 | 1.24 | 541.66 | 479.14 |
| CWT | channel_concat | 175616 | 1.23 | 539.93 | 479.14 |

Thời gian trích xuất toàn cohort theo từng phần ở `embeddings/GAF.json`, `embeddings/CWT.json`.
`channel_mean` và `channel_concat` dùng chung computation, không cộng hai thời gian này như hai extraction độc lập.
`fit_costs.csv` ghi thời gian fit/transform bank từng inner/outer split; `fold_scores.csv` ghi classifier fit và predict.
Bank được tái sử dụng trên các budget trong cùng split, nên thời gian bank không được nhân lặp cho mỗi classifier.
Baseline OOF được đọc lại, chi phí fit bằng 0 ở lần chạy này không có nghĩa baseline không cần train khi dùng thực tế.
Thời gian classifier fit của `original` để trống trong summary vì artifact cũ không tách riêng fit/predict.
Các thời gian CV mới là wall-clock khi chạy tối đa hai fold mỗi nhánh; hai nhánh có thể chạy đồng thời,
chịu ảnh hưởng tranh chấp CPU. Chỉ benchmark GPU chung phía trên là đối chiếu extraction theo cùng workload.

Thời gian mỗi outer job dưới đây bao gồm **toàn bộ 24 ứng viên × (3 inner split + 1 outer fit)**,
không phải thời gian train một cấu hình được chọn. Bảng không tính lại chi phí cũ đã reuse;
không cộng thời gian các job chạy song song để gọi là tổng wall-clock thực nghiệm. Đơn vị giây.

| branch | mode | mean | min | max |
| --- | --- | --- | --- | --- |
| GAF | original | nan | nan | nan |
| GAF | group_adapter | 57.39 | 49.71 | 62.33 |
| GAF | channel_mean | 50.99 | 43.21 | 59.27 |
| GAF | channel_concat | 347.33 | 340.24 | 352.87 |
| CWT | original | nan | nan | nan |
| CWT | group_adapter | 112.88 | 101.44 | 133.74 |
| CWT | channel_mean | 110.60 | 98.68 | 124.30 |
| CWT | channel_concat | 690.85 | 651.85 | 746.10 |

## Kết luận và ứng viên cho bước Fusion sau này

- **GAF:** quy trình chọn trong các embedding mới có cải thiện mô tả trên cùng các outer fold: F1 72.37%, Δ so với gốc +11.42 điểm, Δ so với quy trình giảm chiều cũ +17.10 điểm. Đây là kết quả chọn toàn quy trình, không chứng minh riêng một cấu hình cố định là tối ưu.
- **CWT:** quy trình chọn trong các embedding mới chưa cải thiện F1 so với quy trình gốc: F1 64.98%, Δ so với gốc -3.13 điểm, Δ so với quy trình giảm chiều cũ +0.19 điểm. Đây là kết quả chọn toàn quy trình, không chứng minh riêng một cấu hình cố định là tối ưu.

### Lợi ích đến từ adaptation hay giảm chiều?

- **GAF, tách adaptation khỏi nén:** `channel_concat_raw` có F1 72.65% so với 60.95% của `original_raw` (+11.70 điểm). Khi thêm tìm kiếm reducer, F1 thành 72.70% (+0.05 điểm so với concat raw), AUC thay đổi -1.34 điểm; số chiều từ 75264 xuống trung bình 233.83. Đây là đối chiếu hai quy trình nested; classifier có thể được chọn khác nhau. Để cô lập từng phép biến đổi với classifier cố định, xem ablation matched.
- **CWT, tách adaptation khỏi nén:** `channel_concat_raw` có F1 65.90% so với 68.10% của `original_raw` (-2.20 điểm). Khi thêm tìm kiếm reducer, F1 thành 64.98% (-0.93 điểm so với concat raw), AUC thay đổi -2.16 điểm; số chiều từ 175616 xuống trung bình 208.00. Đây là đối chiếu hai quy trình nested; classifier có thể được chọn khác nhau. Để cô lập từng phép biến đổi với classifier cố định, xem ablation matched.

Với GAF, mức tăng lớn đã xuất hiện ở concat **chưa giảm chiều**; nén chủ yếu giảm kích thước,
trong khi F1 của hai quy trình concat raw/compact gần bằng nhau và AUC có đánh đổi.
Điều này hỗ trợ hướng sửa phép trộn kênh, không phải kết luận PCA luôn làm tăng hiệu năng.
Với CWT, không phương án mới raw/compact nào vượt quy trình gốc về mean F1 trong nested CV.
`all_selected` của CWT cũng kém baseline dù pool có chứa baseline: cho phép `none` thắng
không bảo đảm lựa chọn từ inner CV sẽ đạt điểm outer test cao hơn. Đây là biểu hiện cần lưu ý
của tính bất định khi chọn mô hình trên cohort nhỏ, chưa phải bằng chứng xác định một nguyên nhân duy nhất.

GAF nối riêng kênh là họ đáng ưu tiên cho ablation Fusion tiếp theo; nó được inner CV chọn ở 5/6 fold
trong pool embedding mới. Giữ bản compact theo train fold để giảm kích thước classifier.
GAF adapter theo nhóm là phương án thay thế chi phí encoder gần baseline, nhưng mức F1 thấp hơn.
GAF channel_mean nên giữ như đối chứng cùng chi phí bảy lượt encoder.
CWT giữ embedding gốc làm mốc chính; các bản mới chỉ giữ để truy nguyên/đối chứng, chưa promote.
Các khác biệt GAF/CWT có thể liên quan đến cấu trúc ảnh và độ dư thừa sau chuẩn hóa riêng từng kênh,
nhưng đây mới là giả thuyết cần kiểm chứng, không suy ra nguyên nhân chỉ từ bảng điểm.

Để chọn embedding cố định cho nghiên cứu Fusion tiếp theo, xem thêm `selection_frequency.csv` về mức ổn định
và `extraction_costs.csv` về đánh đổi tài nguyên. `fusion_candidates.csv` phân loại **họ embedding**:
chỉ có Δ F1 nested dương so với gốc mới được ghi là ứng viên cho ablation Fusion sau này, không phải
xác nhận embedding đó cải thiện Fusion. Tần suất chọn lấy từ inner CV của sáu outer fold.

| branch | mode | inner_selected_outer_folds | nested_mean_features | raw_features | inference_time_ratio_vs_original | delta_vs_original_raw_f1_macro_pp | delta_vs_original_compact_f1_macro_pp | recommendation | nested_F1_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAF | group_adapter | 1 | 195.00 | 10752 | 0.99 | 2.70 | 8.37 | candidate_for_future_fusion_ablation | 63.65 |
| GAF | channel_mean | 0 | 1863.50 | 10752 | 6.89 | 10.23 | 15.90 | candidate_for_future_fusion_ablation | 71.18 |
| GAF | channel_concat | 5 | 233.83 | 75264 | 7.21 | 11.75 | 17.42 | candidate_for_future_fusion_ablation | 72.70 |
| CWT | group_adapter | 0 | 12663.17 | 25088 | 0.99 | -1.73 | 1.59 | retain_as_control_only | 66.38 |
| CWT | channel_mean | 0 | 8486.67 | 25088 | 6.58 | -7.73 | -4.41 | retain_as_control_only | 60.38 |
| CWT | channel_concat | 6 | 208.00 | 175616 | 6.56 | -3.13 | 0.19 | retain_as_control_only | 64.98 |

Các cấu hình đứng đầu outer grid chỉ là ứng viên ablation;
nếu lợi ích nested không dương, giữ embedding cũ làm đối chứng thay vì tuyên bố cải tiến.
Không chạy hoặc triển khai Fusion trong nhiệm vụ này.
Ở nghiên cứu Fusion sau, chỉ tái sử dụng cache embedding label-free. PCA/selector phải được fit lại
trong train của split Fusion tương ứng; không lấy tập đặc trưng hợp từ cả sáu outer fold rồi
đánh giá lại trên chính cohort đó như một phép kiểm tra độc lập.

Giữ nguyên baseline Kinematic đầy đủ và MOMENT của đợt trước, không diễn giải việc thay embedding ảnh
là bằng chứng vượt mốc Kinematic. Chênh lệch nhỏ trên cùng 258 người chưa chứng minh ý nghĩa thống kê
hay cải thiện trên external cohort. Đợt thiết kế này tham khảo kết quả cũ trên cùng cohort, nên không
coi đây là một tập xác nhận hoàn toàn độc lập. Sáu outer test fold tách participant nhưng train chồng lấp;
SD fold không phải CI, không dùng t-test độc lập trên sáu fold.

## Artifacts và chạy lại

- `settings.json`, `evaluation_settings.json`, `original_artifacts.json`, `validation.json`.
- `embeddings/`: ma trận mới, checkpoint tiến độ, hash, channel order, chi phí và kiểm chứng baseline.
- `summary.csv`, `fold_scores.csv`, `ablation_matched.csv`, `previous_grid_winners.csv`.
- `nested_summary.csv`, `nested_comparison.csv`, `nested_oof.csv`, `selected_by_fold.csv`, `selection_frequency.csv`.
- `folds/`: train/test IDs, audit từng inner fit, selected feature indices, variance PCA, scores và predictions.
- `selections/`: toàn bộ inner ranking và cấu hình đã chọn theo outer fold.
- `benchmark.json`, `extraction_costs.csv`, `fit_costs.csv`, `cv_job_costs.csv`, `figures/` PNG/SVG và provenance.
- Notebook `notebooks/07_embedding_ablation.ipynb`: bảng số đầy đủ và biểu đồ, không fit mô hình khi mở.

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/extract_embedding_ablation.py --branch GAF
.venv/Scripts/python.exe -X utf8 -B -u scripts/extract_embedding_ablation.py --branch CWT
.venv/Scripts/python.exe -X utf8 -B -u scripts/evaluate_embedding_ablation.py
.venv/Scripts/python.exe -X utf8 -B -u scripts/benchmark_embedding_ablation.py
.venv/Scripts/python.exe -X utf8 -B -u scripts/audit_embedding_baseline.py
.venv/Scripts/python.exe -X utf8 -B scripts/report_embedding_ablation.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_embedding_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 07_embedding_ablation.ipynb --validation-output results/embedding_ablation/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_embedding_ablation.py
```

Extraction và evaluation có checkpoint; thay code/protocol/input sẽ bị từ chối thay vì trộn cache.
Mã bổ sung sử dụng cùng workspace; kết quả/checkpoint/OOF cũ được xác minh SHA-256 không thay đổi.
Tham khảo chính thức: [ViT-B16](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.vit_b_16.html),
[EfficientNet-B4](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.efficientnet_b4.html),
[cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html).
