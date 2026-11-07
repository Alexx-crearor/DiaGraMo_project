# Feature Selection và Dimensionality Reduction — DiaGraMo

Đã hoàn thành **216 cấu hình** = 4 nhánh × (17 cấu hình giảm chiều + baseline) × 3 classifier,
trên cùng **258 participant, Stratified 6-Fold, shuffle=True, seed 42**. Trong đó 12 baseline được đọc
nguyên dự đoán OOF cũ; 204 cấu hình giảm chiều được chạy mới. Thêm **nested CV: 6 outer × 3 inner**
để đánh giá quy trình chọn cả phương pháp/mức chiều/classifier bằng dữ liệu train.
Toàn bộ kết quả cũ và 4 notebook cũ được kiểm tra SHA-256, không bị sửa.

## Thiết lập đã cố định trước khi chạy

- Kinematic: 1.036; MOMENT: 14.336; GAF/ViT-B16: 10.752; CWT/EfficientNet-B4: 25.088 chiều.
- PCA full SVD: **16, 32, 64, 128** thành phần; hoặc giữ **90%, 95%, 99%** phương sai train.
  Không whitening và không chuẩn hóa lại các thành phần sau PCA. Vì train outer có 215 người,
  số thành phần có phương sai khác 0 tối đa 214; train inner có 143–144 người.
- SelectKBest ANOVA `f_classif`: **32, 64, 128, 256, 512** đặc trưng.
- SelectFromModel với ExtraTrees: **32, 64, 128, 256, 512** đặc trưng quan trọng nhất;
  200 cây, min_samples_leaf=2, max_features=sqrt, random_state=42. Cây chỉ dùng nhãn train.
- Logistic Regression, SVM RBF và LightGBM dùng nguyên tham số từ baseline `scripts/evaluate.py`.
  Đây là nghiên cứu 3 classifier đại diện; chưa lặp lại Random Forest, XGBoost, AdaBoost và MLP.
- Metric chính để chọn: **mean F1 macro**; tie-break: Accuracy, AUC, ít chiều hơn, tên cấu hình.
  Không chọn ngưỡng dự đoán hoặc thay nhãn/cohort dựa vào điểm test.

## Không rò rỉ dữ liệu

Mỗi split có một `ReductionBank` riêng. Pipeline median imputation → loại cột hằng →
StandardScaler được fit trên train, sau đó PCA/ANOVA/ExtraTrees cũng chỉ fit trên train.
Một PCA full SVD và một bộ điểm selector được tái sử dụng cho các mức chiều/classifier trên
**cùng train split**; đây chỉ là chia sẻ tính toán, không fit trên toàn cohort. Biến đổi của
outer train tuyệt đối không được tái sử dụng cho inner train.

Hàm chọn nội bộ chỉ nhận `X_train, y_train, participant_ids_train`; không nhận outer test.
Mọi audit inner/outer được kiểm tra tập ID train/test tách rời, coverage và việc chọn đúng
theo điểm inner-CV. Với PCA 90/95/99%, số chiều được xác định riêng từ phương sai train.

## Cấu hình cố định tốt nhất trong bảng thăm dò

Đơn vị **%**; mỗi bộ ba là **Accuracy / F1 macro / ROC-AUC**, đều là **trung bình 6 fold**.
`summary.csv` giữ thêm SD và OOF gộp. AUC gộp có thể khác AUC trung bình.
Các cấu hình dưới đây được xếp hạng trên chính 6 fold nên có **thiên lệch chọn cấu hình**;
không xem điểm của cấu hình thắng bảng này là ước lượng độc lập sau tối ưu.
Baseline cũ tốt nhất được chọn theo cùng tiêu chí F1 macro trong toàn bộ 7 classifier cũ.

| Nhánh | Cấu hình giảm chiều | Chiều gốc → mới | Baseline tốt nhất cũ (Acc/F1/AUC) | Mới (Acc/F1/AUC) | Δ F1 so với baseline cũ (đpt) |
| --- | --- | --- | --- | --- | --- |
| Kinematic | extratrees_512 + LightGBM | 1036 → 512.0 | LightGBM: 84.88 / 84.70 / 90.40 | 82.95 / 82.70 / 90.58 | -2.00 |
| MOMENT | anova_64 + LogisticRegression | 14336 → 64.0 | SVM: 67.83 / 64.79 / 75.07 | 71.71 / 71.36 / 75.59 | 6.57 |
| GAF | anova_256 + SVM | 10752 → 256.0 | LightGBM: 63.18 / 61.65 / 66.30 | 67.05 / 66.00 / 66.17 | 4.35 |
| CWT | pca_var99 + LogisticRegression | 25088 → 203.3 | LogisticRegression: 68.60 / 68.10 / 75.85 | 71.32 / 70.74 / 76.07 | 2.63 |

Đối chiếu từng cấu hình với baseline **cùng classifier** nằm ở
[comparison_same_classifier.csv](comparison_same_classifier.csv); không gán lợi ích đổi classifier cho giảm chiều.

## So sánh phương pháp tốt nhất theo nhánh

| Nhánh | Phương pháp | Cấu hình | Acc | F1 macro | AUC |
| --- | --- | --- | --- | --- | --- |
| Kinematic | extratrees | extratrees_512 + LightGBM | 82.95 | 82.70 | 90.58 |
| Kinematic | anova | anova_256 + SVM | 82.17 | 81.61 | 87.37 |
| Kinematic | pca | pca_var90 + SVM | 79.84 | 79.25 | 86.85 |
| MOMENT | anova | anova_64 + LogisticRegression | 71.71 | 71.36 | 75.59 |
| MOMENT | pca | pca_var99 + SVM | 70.54 | 70.29 | 75.29 |
| MOMENT | extratrees | extratrees_512 + SVM | 70.54 | 69.50 | 75.73 |
| GAF | anova | anova_256 + SVM | 67.05 | 66.00 | 66.17 |
| GAF | pca | pca_16 + LogisticRegression | 65.89 | 64.52 | 70.79 |
| GAF | extratrees | extratrees_128 + SVM | 62.40 | 60.68 | 61.87 |
| CWT | pca | pca_var99 + LogisticRegression | 71.32 | 70.74 | 76.07 |
| CWT | anova | anova_512 + LogisticRegression | 67.83 | 67.13 | 71.11 |
| CWT | extratrees | extratrees_32 + LogisticRegression | 67.05 | 66.22 | 71.58 |

## Kiểm tra bằng nested CV — ưu tiên bảng này để kết luận về tối ưu

Ở mỗi outer fold, inner Stratified 3-Fold trên 215 người train chọn trong **54 ứng viên**
(18 reducer kể cả `none` × 3 classifier). Đối chứng chọn trong 3 classifier giữ nguyên chiều,
cũng chỉ dựa trên inner CV. Hai lựa chọn được đánh giá trên cùng 43 người outer test.
Nhánh `none` được phép thắng nếu giảm chiều không có lợi; không bắt buộc chọn một reducer.
Mỗi outer fold có thể chọn cấu hình khác nhau: bảng này đánh giá **quy trình chọn**, không xác nhận
một cấu hình cố định là tối ưu trên dữ liệu mới. 3-fold inner được dùng thêm, còn giao thức
đánh giá outer 6-fold, seed và participant giữ nguyên baseline.

| Nhánh | Chọn classifier, giữ nguyên chiều (Acc/F1/AUC) | Chọn cả reducer và classifier (Acc/F1/AUC) | Δ F1 (đpt) | Chiều TB sau chọn |
| --- | --- | --- | --- | --- |
| Kinematic | 80.23 / 80.00 / 87.48 | 79.07 / 78.83 / 85.43 | -1.17 | 444.33 |
| MOMENT | 65.12 / 63.47 / 70.11 | 67.05 / 66.18 / 71.60 | 2.71 | 225.33 |
| GAF | 62.40 / 60.95 / 66.27 | 56.59 / 55.28 / 61.05 | -5.67 | 133.33 |
| CWT | 68.60 / 68.10 / 75.85 | 65.89 / 64.79 / 70.45 | -3.31 | 216.67 |

Lựa chọn từng fold: [selected_by_fold.csv](selected_by_fold.csv).
Tần suất chọn: [selection_frequency.csv](selection_frequency.csv).

## Phương pháp nào tốt nhất và cách giải thích

- **Kinematic:** cấu hình giảm chiều đứng đầu bảng thăm dò là `extratrees_512 / LightGBM` (512.0/1036 chiều, giảm 50.58%). F1 macro 82.70% (SD 3.80 điểm); chênh baseline cùng classifier -2.00 điểm và baseline cũ tốt nhất -2.00 điểm. Quy trình chọn trong train không cải thiện F1 trong đánh giá nested (-1.17 điểm).
- **MOMENT:** cấu hình giảm chiều đứng đầu bảng thăm dò là `anova_64 / LogisticRegression` (64.0/14336 chiều, giảm 99.55%). F1 macro 71.36% (SD 5.78 điểm); chênh baseline cùng classifier +9.24 điểm và baseline cũ tốt nhất +6.57 điểm. Quy trình chọn trong train cải thiện F1 trong đánh giá nested (+2.71 điểm).
- **GAF:** cấu hình giảm chiều đứng đầu bảng thăm dò là `anova_256 / SVM` (256.0/10752 chiều, giảm 97.62%). F1 macro 66.00% (SD 5.82 điểm); chênh baseline cùng classifier +5.31 điểm và baseline cũ tốt nhất +4.35 điểm. Quy trình chọn trong train không cải thiện F1 trong đánh giá nested (-5.67 điểm).
- **CWT:** cấu hình giảm chiều đứng đầu bảng thăm dò là `pca_var99 / LogisticRegression` (203.3/25088 chiều, giảm 99.19%). F1 macro 70.74% (SD 6.75 điểm); chênh baseline cùng classifier +2.63 điểm và baseline cũ tốt nhất +2.63 điểm. Quy trình chọn trong train không cải thiện F1 trong đánh giá nested (-3.31 điểm).

Trong bảng thăm dò, số nhánh đứng đầu theo phương pháp: **{'anova': 2, 'pca': 1, 'extratrees': 1}**. Đây là mô tả kết quả
trên cohort hiện tại, không phải bằng chứng một phương pháp luôn tốt nhất.

**Khuyến nghị từ đợt chạy này:** giữ baseline đầy đủ chiều của Kinematic nếu ưu tiên F1 macro.
MOMENT là nhánh có bằng chứng mô tả tích cực nhất cho việc tiếp tục tối ưu: ANOVA-64 + Logistic
Regression đứng đầu bảng cố định và quy trình chọn bằng inner CV cũng tăng F1 trên outer test.
Tuy nhiên, inner CV không chọn ANOVA-64 ở mọi fold; mức tăng nested đánh giá toàn bộ quy trình
chọn, chưa xác nhận riêng cấu hình ANOVA-64. Với GAF và CWT, giữ baseline làm cấu hình tham chiếu
cho đến khi có đánh giá xác nhận thêm, vì lợi ích của cấu hình thắng bảng không lặp lại trong nested CV.

Một cách giải thích **cần kiểm chứng** là ANOVA loại nhiều tọa độ MOMENT ít phân biệt hai nhóm,
còn PCA gộp các tọa độ CWT tương quan. Đây là cơ chế phù hợp với phương pháp, chưa phải nguyên nhân
đã chứng minh bằng thực nghiệm ablation. Số chiều rất lớn so với 215 người train và việc inner CV
chọn các cấu hình khác nhau giữa fold cho thấy cần thận trọng với tính ổn định của cấu hình thắng.

- **PCA** loại dư thừa tuyến tính bằng cách xoay sang các trục phương sai lớn. Nó hữu ích khi
  embedding có nhiều tọa độ tương quan, nhưng không dùng nhãn nên có thể bỏ hướng phương sai nhỏ
  chứa tín hiệu phân lớp. Giữ 99% phương sai không đồng nghĩa giữ 99% thông tin chẩn đoán.
- **ANOVA** giữ tọa độ có khác biệt trung bình giữa hai nhóm, dễ diễn giải hơn PCA đối với động học.
  Nó không mô hình hóa tương tác và có thể giữ nhiều biến tương quan. Trong p≫n, điểm univariate
  dễ biến động; vì vậy chỉ chọn trong train và có nested CV.
- **ExtraTrees** xếp hạng theo phân chia cây, có khả năng khai thác quan hệ phi tuyến/tương tác.
  Với chỉ 215 người train và hàng chục nghìn tọa độ, importance có thể phân tán giữa các biến
  tương quan; không diễn giải importance như quan hệ nhân quả hay bằng chứng lâm sàng.
- [feature_stability.csv](feature_stability.csv) ghi Jaccard trung bình giữa 15 cặp outer-fold
  cho mỗi selector/mức chiều. Các tập train chồng lấp nên đây chỉ là chỉ báo độ ổn định mô tả.
- Giảm chiều giảm kích thước đầu vào classifier, **không** giảm số chiều cần trích từ encoder trước
  reducer. Không tuyên bố tăng tốc end-to-end vì chưa benchmark toàn pipeline có cùng điều kiện.

Khác biệt điểm nhỏ trên 258 người chưa chứng minh ý nghĩa thống kê hoặc khả năng tổng quát hóa.
SD giữa 6 fold không phải khoảng tin cậy; không dùng kiểm định độc lập sai cho các train fold chồng lấp.
Các giả định nhãn `group`, bộ đặc trưng dựng lại và encoder của thực nghiệm cũ vẫn giữ nguyên;
chưa có tập kiểm tra ngoài và chưa có đánh giá chẩn đoán lâm sàng.

## Tệp kết quả và chạy lại

- [summary.csv](summary.csv): đủ 216 cấu hình; mean, SD, OOF gộp, số chiều và phương sai PCA.
- [best_per_branch.csv](best_per_branch.csv), [best_by_method.csv](best_by_method.csv).
- [nested_comparison.csv](nested_comparison.csv), [nested_summary.csv](nested_summary.csv),
  [nested_oof.csv](nested_oof.csv): đánh giá quy trình chọn bằng train-CV.
- [fold_scores.csv](fold_scores.csv), [oof/](oof/): điểm và dự đoán từng người của mọi cấu hình cố định.
- [folds/](folds/): fit participant IDs, selected feature indices, phương sai PCA, toàn bộ điểm inner-CV.
- [settings.json](settings.json), [original_artifacts.json](original_artifacts.json),
  [validation.json](validation.json): thiết lập, SHA-256 và kiểm tra tính toàn vẹn.

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/optimize_features.py --workers 2
```

Chạy lại cùng lệnh sẽ tiếp tục từ các outer job đã hoàn tất; nếu đổi tham số, mã thực nghiệm hoặc
đặc trưng, phải dùng `--output results/feature_optimization_<ten_moi>` để không trộn kết quả.
Tạo lại báo cáo từ kết quả đã chạy:

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/optimize_features.py --report-only
```

Tham khảo kỹ thuật: [scikit-learn 1.7 feature selection](https://scikit-learn.org/1.7/modules/feature_selection.html),
[PCA](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.PCA.html),
[nested CV](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
