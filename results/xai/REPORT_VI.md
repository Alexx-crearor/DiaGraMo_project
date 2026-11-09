# Explainable AI — Kinematic và Fusion Kinematic+CWT

## Kết quả chính và phạm vi

Giải thích **mô hình đã đánh giá**, không thay kiến trúc, classifier, reducer, trọng số hoặc threshold.
Giữ **258 participant, nhãn group, sáu outer folds và seed 42**. Attribution chỉ tính trên 43 outer Test
của mô hình fit trên 215 outer Train tương ứng. Baseline OOF và hai Fusion được replay khớp trước giải thích.
Trạng thái cuối cùng chỉ lấy từ `validation.json complete=true`, sau tests, notebook và hình.

Theo attribution mass của raw TreeSHAP OOF, ba tác vụ đứng đầu là **copy, dictation, lower-loops-fast**,
chiếm **54.89%** tổng mean absolute attribution. Ba nhóm có tổng mass cao nhất là **FFT, Statistics, Trajectory**.
Đây là mô tả việc mô hình sử dụng thông tin, không phải bằng chứng nhân quả, biomarker hay chẩn đoán lâm sàng.

## Mapping thực tế và preprocessing

`feature_mapping.csv` ánh xạ đủ 1.036 cột = 14 tác vụ × 74 đặc trưng. Thứ tự task lấy từ
`data/derived/schema.json` và kiểm tra bằng `prepare.py TASKS`; tên/thứ tự feature kiểm tra trực tiếp
từ `features.py kinematic()` trên raw source. Tái dựng đủ 14 task của participant đầu, giữa, cuối
(chọn không dùng nhãn) và đối chiếu toàn bộ 1.036 giá trị cache; xem `mapping_audit.json`.
Không giả định tên cột từ vị trí hoặc tự đổi thứ tự tác vụ. Metadata, cognitive scores và ID không là feature.

Nhóm lấy theo **12 prefix thực tế trong code**: Velocity, Acceleration, Jerk, Peaks, Lifts,
Pressure, Orientation, Trajectory, Curvature, Statistics, Energy, FFT. Statistics giữ các entropy/
skewness/kurtosis của nhiều tín hiệu; Energy giữ RMS; không đếm lặp các feature này vào Pressure/Velocity.
Đây là taxonomy của implementation, không ép thành số nhóm được mô tả trong bài báo gốc.

SHAP chạy trên đầu vào sau median imputer → bỏ cột hằng → StandardScaler đã fit trong outer Train.
Ánh xạ qua support của VarianceThreshold để trả về đúng 1.036 tọa độ; cột bị bỏ có SHAP=0.
Không inverse-transform SHAP bằng scaler: attribution vẫn ở output space; giá trị X trong hình là
giá trị raw để đọc đơn vị đầu vào. Scaler affine theo từng cột không làm đổi danh tính feature.

## SHAP output space và kiểm tra cộng dồn

SHAP **0.49.1**, tương thích Python 3.10 của môi trường hiện có; NumPy/sklearn/LightGBM cũ không bị nâng.
Tính bằng `TreeExplainer(...)(X_test)` và API `Explanation`, `approximate=False`.

- **Primary:** `tree_path_dependent`, `model_output=raw`. Reference là training cover/leaf counts
  của chính model outer Train, không lấy Test làm background.
- **Sensitivity:** `interventional raw` và `interventional probability`, cùng 100 background rows
  lấy ngẫu nhiên seed 42 **chỉ từ outer Train**, không stratify hoặc chọn theo nhãn/SHAP.
- Kiểm tra cả 258 người: `base_raw + Σφ_raw = predict(raw_score=True)`;
  với LightGBM binary này, đã xác minh raw score = `log(p/(1−p))` và sigmoid(raw)=p(group=1).
- Với probability SHAP: `base_probability + Σφ_probability = predict_proba[:,1]`.
  Probability SHAP có thể âm; dấu dương đẩy về group=1, không đồng nghĩa "gây bệnh".
- **Không lấy sigmoid từng φ_raw rồi cộng** để thành probability attribution. Kể cả cùng background,
  `E[sigmoid(raw)]` thường khác `sigmoid(E[raw])`; bảng `output_space_checks.csv` giữ hai reference
  để kiểm tra. Primary raw/path và probability/interventional còn khác giả định masking nên không
  diễn giải khác biệt giữa chúng như chỉ do hàm sigmoid.

**Kiểm chứng backend:** SHAP 0.49 independent engine lưu threshold trong C++ `Node.thres` ở float32.
Lượt thử `attempts/precision_02/` phát hiện residual raw ~6.46e−4 và probability ~7.78e−5 ở fold 2,
dù kiểm tra mặc định của SHAP vẫn qua. Không tăng tolerance hoặc cộng offset để ép khớp.
Thay vào đó chỉ dùng cho **explainer copy** một lossless partition encoding: với mỗi feature, lấy
các split thresholds float64 từ **frozen model**, ánh xạ input vào bin theo `searchsorted(side=left)`;
thay threshold thứ j bởi j+0.5. Mọi phép so sánh x≤t tương đương bin(x)≤j+0.5, kể cả đúng tại t.
Giữ nguyên leaf values, cây, feature players và missing routing; checkpoint/pipeline gốc không đổi.
Do phép biến đổi theo từng cột bảo toàn mọi split, mọi feature-masked coalition có cùng prediction,
nên game interventional không đổi. Cuts chỉ đến từ model đã fit Train, không học từ Test hay nhãn.
Unit test kiểm tra threshold sát ranh giới và SHAP của stump bằng tính trực tiếp; mỗi fold kiểm tra
toàn bộ split boundaries, Train background, Test và 300 hybrid masks đều khớp raw model tới 1e−12.
Attribution sau đó vẫn phải qua tolerance 2e−6 và kiểm tra probability/raw độc lập.

Attribution chịu ảnh hưởng của tương quan feature và lựa chọn reference/masking; interventional không
biến dữ liệu quan sát thành bằng chứng nhân quả. Kết quả độ nhạy thứ hạng:

| level | raw_path_vs_interventional_raw_spearman | interventional_raw_vs_probability_spearman |
| --- | --- | --- |
| feature | 0.9994 | 0.9994 |
| task | 1.0000 | 0.9956 |
| group | 1.0000 | 0.9860 |

## Top Features — thang log-odds

`sum_mean_abs_shap` ở hàng feature là mean |φ| trên 258 outer Test rows; `top_k_fold_count`
là số fold feature thuộc Top-20. Cột này chỉ mô tả ổn định, không dùng chọn lại feature/model.

| task | feature | group | sum_mean_abs_shap | top_k_fold_count |
| --- | --- | --- | --- | --- |
| copy | trajectory_width | Trajectory | 0.9669 | 6 |
| copy | fft_pressure_dominant_hz | FFT | 0.7098 | 6 |
| copy | statistics_acc_skew | Statistics | 0.4661 | 6 |
| copy | trajectory_duration | Trajectory | 0.4185 | 6 |
| copy | trajectory_bbox_area | Trajectory | 0.3863 | 6 |
| RCFT-recall | statistics_acc_kurtosis | Statistics | 0.3580 | 6 |
| lower-loops-precise | fft_y_bandwidth_hz | FFT | 0.2559 | 6 |
| dictation | orientation_azimuth_std | Orientation | 0.2518 | 6 |
| lower-loops-fast | pressure_std | Pressure | 0.2423 | 5 |
| dictation | energy_y_rms | Energy | 0.2369 | 5 |
| copy | fft_y_bandwidth_hz | FFT | 0.2164 | 5 |
| small-spiral-fast | statistics_pressure_skew | Statistics | 0.1557 | 4 |
| lower-loops-precise | fft_speed_centroid_hz | FFT | 0.1452 | 3 |
| small-spiral-fast | fft_speed_centroid_hz | FFT | 0.1361 | 2 |
| lower-loops-fast | orientation_tilt_std | Orientation | 0.1304 | 2 |
| small-spiral-precise | fft_x_bandwidth_hz | FFT | 0.1283 | 2 |
| RCFT-recognition | statistics_acc_skew | Statistics | 0.1199 | 3 |
| copy | fft_y_centroid_hz | FFT | 0.1087 | 3 |
| dictation | jerk_std | Jerk | 0.1040 | 1 |
| copy | jerk_std | Jerk | 0.0977 | 1 |

## Top Feature Groups

| name | n_features | sum_mean_abs_shap | per_feature_mean_abs | attribution_mass_share_percent |
| --- | --- | --- | --- | --- |
| FFT | 224 | 3.4003 | 0.0152 | 27.9621 |
| Statistics | 140 | 2.6258 | 0.0188 | 21.5927 |
| Trajectory | 112 | 2.4655 | 0.0220 | 20.2745 |
| Orientation | 56 | 1.0393 | 0.0186 | 8.5462 |
| Pressure | 112 | 0.5890 | 0.0053 | 4.8435 |
| Acceleration | 84 | 0.4734 | 0.0056 | 3.8929 |
| Energy | 42 | 0.4528 | 0.0108 | 3.7232 |
| Jerk | 42 | 0.4119 | 0.0098 | 3.3870 |
| Peaks | 42 | 0.2777 | 0.0066 | 2.2832 |
| Velocity | 112 | 0.1573 | 0.0014 | 1.2937 |
| Curvature | 28 | 0.1396 | 0.0050 | 1.1482 |
| Lifts | 42 | 0.1280 | 0.0030 | 1.0526 |

Tổng mass có xu hướng lớn hơn khi nhóm có nhiều cột. Vì vậy giữ thêm `per_feature_mean_abs` và
số cột; không suy rằng FFT quan trọng sinh học hơn chỉ vì có nhiều tọa độ.
Hai định nghĩa được lưu riêng: **mass = E[Σ|φ_j|]** và **net magnitude = E[|Σφ_j|]**.
Signed sum theo group/task vẫn cộng đúng về output − base; mass không có tính cộng dồn về prediction.
Đây là **aggregation của individual SHAP**, không phải Group SHAP/Owen values tính bằng joint masker.

## Top Tasks và độ ổn định sáu fold

| name | sum_mean_abs_shap | attribution_mass_share_percent | mean_rank | sd_rank | top_k_fold_count |
| --- | --- | --- | --- | --- | --- |
| copy | 4.2977 | 35.3414 | 1.0000 | 0.0000 | 6 |
| dictation | 1.3630 | 11.2088 | 2.3333 | 0.5164 | 6 |
| lower-loops-fast | 1.0142 | 8.3401 | 3.6667 | 1.0328 | 2 |
| small-spiral-fast | 0.8941 | 7.3521 | 4.3333 | 0.8165 | 1 |
| RCFT-recall | 0.8687 | 7.1436 | 4.8333 | 1.9408 | 2 |
| RCFT-recognition | 0.6979 | 5.7388 | 5.8333 | 2.0412 | 1 |
| lower-loops-precise | 0.6822 | 5.6096 | 6.6667 | 1.3663 | 0 |
| small-spiral-precise | 0.4970 | 4.0873 | 8.8333 | 1.9408 | 0 |
| big-spiral-fast | 0.3785 | 3.1127 | 9.8333 | 1.4720 | 0 |
| rainbow-fast | 0.3078 | 2.5312 | 10.6667 | 3.0768 | 0 |
| upper-loops-fast | 0.2992 | 2.4603 | 11.6667 | 1.6330 | 0 |
| combined-loops-fast | 0.2964 | 2.4376 | 11.8333 | 1.7224 | 0 |
| saw-fast | 0.2908 | 2.3915 | 11.6667 | 2.1602 | 0 |
| combined-loops-precise | 0.2730 | 2.2450 | 11.8333 | 1.7224 | 0 |

Thứ hạng theo tổng mass; mỗi tác vụ có cùng 74 feature nên tránh khác biệt số cột giữa task.
`mean_rank ± sd_rank`, Top-3 frequency và heatmap sáu fold phản ánh cả thay đổi model lẫn 43 Test rows.
Train folds chồng lấp; SD không phải CI, 15 cặp fold không phải 15 mẫu độc lập.

| level | mean_spearman | min_spearman | max_spearman | mean_top_k_jaccard |
| --- | --- | --- | --- | --- |
| feature | 0.3149 | 0.2777 | 0.3594 | 0.4161 |
| task | 0.8221 | 0.7231 | 0.9253 | 0.5667 |
| group | 0.9133 | 0.8322 | 0.9650 | 1.0000 |

## Grouped permutation trên held-out Test

Hoán vị đồng thời toàn bộ 74 cột của một task bằng cùng một permutation người; làm thêm 12 block
feature group xuyên các task. **30 lần/block/fold, 26 block × 30 × 6 = 4.680** đánh giá prediction.
Giữ model/preprocessing cố định, không fit lại và không chọn threshold. Chỉ dùng Test labels để tính
chênh F1/Accuracy/AUC/Balanced Accuracy. Mọi permutation và prediction được lưu, replay lại trong validator.

Đơn vị **điểm phần trăm**: điểm gốc trừ điểm sau hoán vị. Bảng là mean sáu fold (mỗi fold đã lấy mean
30 lần); SD là giữa sáu fold, không gộp 180 lần như mẫu độc lập. Giữ cả giá trị âm.

| block | mean_drop_f1_macro_pp | fold_sd_drop_f1_macro_pp | mean_drop_auc_pp |
| --- | --- | --- | --- |
| copy | 23.0300 | 3.5122 | 22.9008 |
| lower-loops-fast | 3.6766 | 2.8080 | 2.3394 |
| RCFT-recognition | 2.8633 | 1.7258 | 0.4942 |
| dictation | 2.5003 | 3.8567 | 1.2302 |
| RCFT-recall | 2.4695 | 2.0067 | 0.6768 |
| lower-loops-precise | 2.1038 | 2.8100 | 0.5673 |
| small-spiral-fast | 2.0664 | 2.8513 | 0.6656 |
| rainbow-fast | 1.2082 | 1.4640 | 0.2695 |
| big-spiral-fast | 0.9726 | 0.7994 | 0.2103 |
| small-spiral-precise | 0.8009 | 1.8584 | 0.4415 |
| saw-fast | 0.7408 | 0.8188 | -0.4273 |
| upper-loops-fast | 0.4219 | 0.4539 | 0.1665 |
| combined-loops-fast | 0.3355 | 0.8287 | -0.1122 |
| combined-loops-precise | 0.1029 | 1.7768 | -0.1100 |

Đối chiếu thứ hạng SHAP mass và permutation F1:

| level | shap_mass_vs_permutation_f1_spearman |
| --- | --- |
| task | 0.9165 |
| group | 0.8811 |

Đây là marginal/joint-block permutation: bảo toàn cấu trúc trong block nhưng phá quan hệ với các
block còn lại, có thể tạo tổ hợp ngoài phân phối. Tương quan/redundancy và chỉ 43 người/fold có thể
làm SHAP và permutation khác nhau; không chọn chỉ metric nào thuận lợi hoặc coi permutation là nhân quả.

**Chưa thể kết luận số tác vụ tối thiểu.** Các tác vụ mass thấp vẫn có thể bổ sung cho nhau;
không cộng Top-k mass thành accuracy giữ lại. Đề xuất một task ablation riêng: chọn subset hoặc
xếp hạng task trong inner Train, refit toàn pipeline trên subset trong mỗi split, đối chiếu cả
accuracy/F1/AUC và thời gian làm bài trên outer Test, xác nhận bằng cohort ngoài. Không chạy thí nghiệm
giảm task hoặc tối ưu accuracy trong nhiệm vụ XAI này.

## Fusion: xác suất, trọng số và confidence gate

`fusion_participants.csv` có đủ **516 hàng** (258 × Late/Adaptive): pK, pC, pFusion, native predictions,
effective weights, nominal weights, gate status/threshold/budget, Δp do CWT và outcome.
Tất cả probability được tái tạo từ đúng branch checkpoint/combiner rồi đối chiếu `nested_oof.csv`.

- **Late:** pFusion = wK pK + wC pC, wK+wC=1. Không có confidence gate;
  gate status ghi `not_applicable`. Δp_CWT = wC(pC−pK).
- **Adaptive:** g=1[|pK−0.5|≤τ]; pFusion=pK+gα(pC−pK). Có một nhánh phụ nên auxiliary simplex
  weight bằng 1; α và τ đã được chọn trong inner CV từ nghiên cứu trước. Effective wC=gα,
  effective wK=1−gα. Đây là **confidence-gated probability correction**, **không phải learned attention network**.
  Khi g=0, pFusion=pK đúng số học. 37/258 participant có gate active.
- Δp là thay đổi xác suất **so với Kinematic**, không phải SHAP hay causal contribution của một modality.
  Không nhân raw Kinematic SHAP với một weight rồi gọi là probability SHAP của Fusion. Gate phụ thuộc
  pK tạo tương tác/điểm chuyển trạng thái; giải thích ở đây là replay cơ chế thực tế.

| scenario | corrected | harmed | unchanged_correct | unchanged_wrong |
| --- | --- | --- | --- | --- |
| adaptive__KC | 8 | 6 | 213 | 31 |
| late__KC | 9 | 8 | 211 | 30 |

Late sửa đúng 9 người và làm sai 8 người (ròng +1); Adaptive sửa đúng 8, làm sai 6 (ròng +2).
Các con số khớp nghiên cứu Fusion trước, không phải kết quả mô hình mới.
Trên toàn cohort, pC không tự giúp nếu quá yếu hoặc kéo một pK đúng qua ngưỡng 0.5.
`fusion_confidence_bins.csv` và scatter Δp cho thấy khi nào correction thay nhãn hoặc chỉ thay độ tự tin.
Đây là phân tích hậu nghiệm, không chỉnh gate theo các ca Test vừa quan sát.

Tham số/nhánh CWT thực tế từng fold:

| scenario | fold | cwt_reducer | cwt_dimensions | mean_effective_cwt_weight | gate_threshold | budget | mean_abs_delta_cwt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| late__KC | 0 | none | 25088 | 0.2253 | nan | nan | 0.0865 |
| adaptive__KC | 0 | pca_var99 | 204 | 0.0698 | 0.1000 | 0.5000 | 0.0199 |
| late__KC | 1 | pca_var99 | 203 | 0.2839 | nan | nan | 0.0829 |
| adaptive__KC | 1 | pca_var99 | 203 | 0.0233 | 0.1000 | 0.2500 | 0.0088 |
| late__KC | 2 | none | 25088 | 0.2753 | nan | nan | 0.0919 |
| adaptive__KC | 2 | none | 25088 | 0.0930 | 0.2500 | 0.5000 | 0.0451 |
| late__KC | 3 | none | 25088 | 0.2077 | nan | nan | 0.0595 |
| adaptive__KC | 3 | pca_var99 | 203 | 0.0116 | 0.1000 | 0.2500 | 0.0054 |
| late__KC | 4 | none | 25088 | 0.3003 | nan | nan | 0.1099 |
| adaptive__KC | 4 | pca_var99 | 204 | 0.0581 | 0.2500 | 0.2500 | 0.0221 |
| late__KC | 5 | pca_var99 | 203 | 0.2014 | nan | nan | 0.0759 |
| adaptive__KC | 5 | pca_var99 | 203 | 0.0814 | 0.2500 | 0.5000 | 0.0388 |

Baseline Kinematic Accuracy OOF 84.88%, F1 84.72%; Late KC 85.27%/85.10%; Adaptive KC 85.66%/85.56%.
Mức tăng nhỏ và KC là family đứng đầu sau xem outer results; `all_selected` trước đó chỉ đạt
81.78%/81.52%. XAI không làm tăng bằng chứng tổng quát hóa, không dùng để thay baseline tự động.

## CWT/PCA: giới hạn diễn giải

CWT ở hai pipeline dùng **LogisticRegression**, không dùng TreeExplainer cho model này.
Lưu exact independent-reference linear decomposition: φ_j = β_j(z_j−mean_background_train(z_j));
base = intercept + β·mean_background, cộng về **CWT log-odds**; sigmoid tổng khớp pC.
Phép trừ/mean attribution tính float64 để không mất đẳng thức với input cache float32.
Lượt kiểm tra ban đầu phát hiện lỗi cỡ 8.5e−8 do float32 subtraction; giữ ở
`attempts/precision_01/` với trạng thái chưa hoàn tất, không nới tolerance để che lỗi.

PCA attribution chỉ có nghĩa ở **tọa độ embedding/PCA của model đó**. `cwt_component_importance.csv`
giữ fold/scenario/reducer và component index riêng; **không gộp PC1 giữa các fold**, vì PCA axes/signs
được fit riêng. Với `none`, tên là preprocessed embedding coordinate, không phải đặc trưng sinh học.
Không back-project attribution qua encoder, không gán PC cho task/tín hiệu, không diễn giải thành
thang tần số CWT hoặc ý nghĩa sinh học khi chưa có mapping attribution hợp lệ.

## Hình/bảng đáng giữ để viết paper

- **Main:** SHAP beeswarm/bar (raw log-odds), task importance với fold SD, task rank heatmap,
  grouped permutation tasks và Fusion corrected/harmed + Δp scatter.
- **Supplement:** Summary plot (cùng dữ liệu với beeswarm, không phải bằng chứng độc lập),
  dependence scatter, bốn waterfall cho corrected/harmed/unchanged correct/unchanged wrong,
  group mass so với per-feature average, task×group heatmap, task-level waterfall và CWT PCA bar.
- Hình dependence là quan hệ X–SHAP quan sát qua sáu mô hình, không phải partial dependence,
  can thiệp nhân quả hay khuyến nghị thay đổi hành vi. Waterfall lấy participant đầu theo ID của
  từng outcome, chỉ để minh họa; `local_cases.json` ghi rule và fold, không chọn ca có hình đẹp nhất.

Không kết luận marker bệnh, tối thiểu bao nhiêu task hay tính an toàn lâm sàng từ cohort này.
Giải thích mô hình phụ thuộc bộ feature dựng lại và nhãn `group` (khác diagnosis flag cụ thể).
Cần ablation có thiết kế và external validation riêng cho kết luận về task reduction/tổng quát hóa.

## Artifacts và chạy lại

1288 artifact cũ được kiểm tra SHA-256, không bị sửa. Tất cả kết quả mới ở `results/xai/`;
notebook mới `notebooks/09_xai.ipynb`, code `scripts/*xai*.py`, module `diagramo/explain.py`, tests riêng.
`attributions/` giữ ma trận SHAP 258×1036 và linear CWT contributions; `folds/` giữ participant IDs,
background IDs, errors, SHA và cơ chế; `permutation/` giữ mọi permutation/prediction; CSV cấp feature,
task/group/fold/outcome phục vụ viết paper. Không có attribution từ model train full cohort.

```powershell
.venv/Scripts/python.exe -m pip install --no-deps -r requirements-xai.txt
.venv/Scripts/python.exe -X utf8 -B -u scripts/analyze_xai.py
.venv/Scripts/python.exe -X utf8 -B -u scripts/report_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/plot_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_xai_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 09_xai.ipynb --validation-output results/xai/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_xai.py
```

Tham khảo chính thức: [TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html),
[LinearExplainer](https://shap.readthedocs.io/en/stable/generated/shap.LinearExplainer.html),
[Permutation importance](https://scikit-learn.org/1.7/modules/permutation_importance.html).
