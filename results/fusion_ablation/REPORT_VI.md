# Multimodal Fusion — Nested CV và Ablation Study

## Kết quả và trạng thái

**Có cải thiện mô tả ở cả F1 trung bình fold và Accuracy OOF so với Kinematic.** Cấu hình Fusion có mean F1 cao nhất trong bốn phương pháp/tổ hợp chính là
`adaptive__KC`: OOF Accuracy 85.66%, F1 macro 85.56%,
ROC-AUC 91.16%; Δ OOF F1 +0.85 điểm phần trăm.
Đây là hàng đứng đầu sau khi xem kết quả, có thiên lệch chọn hàng thắng. `all_selected` là quy trình
chọn giữa các phương pháp/tổ hợp chính hoàn toàn bằng inner CV, cần đọc cùng hàng này.

Baseline chính **Kinematic nguyên chiều + LightGBM**, giữ nguyên tham số cũ:
OOF Accuracy **84.88%**, F1 macro **84.72%**,
ROC-AUC **90.69%**. Dự đoán và xác suất tái fit khớp baseline cũ trong tolerance 1e-12.
Kết quả mới không thay thế baseline. Chỉ `validation.json complete=true` xác nhận hoàn tất sau
kiểm tra notebook, hình, checkpoint, OOF, nguồn dữ liệu và tests.

## Thiết kế cố định trước khi chạy

258 participant duy nhất, nhãn `group`, outer Stratified 6-Fold với assignments cũ, seed 42;
inner Stratified 3-Fold chọn theo mean F1 macro, sau đó Accuracy, ROC-AUC, ít chiều và ID cấu hình.
Không chọn threshold; giữ ngưỡng xác suất 0.5 cho bộ kết hợp. Mô hình đơn và Early dùng predict()
nguyên bản (SVM predict có thể khác threshold predict_proba). Không chọn dựa outer Test.

Các tổ hợp chính: KM=Kinematic+MOMENT, KG=Kinematic+GAF, KC=Kinematic+CWT,
KMG=Kinematic+MOMENT+GAF. Hai đối chứng KG_adapter/KMG_adapter dùng GAF group_adapter.
Không mở rộng sang tổ hợp khác. Tất cả cache embedding label-free giữ nguyên; GPU không được sử dụng,
không trích xuất hoặc fine-tune encoder, không huấn luyện trên toàn cohort để tạo OOF.

- **Original Early:** embedding gốc MOMENT/GAF/CWT, nối vector và pipeline cũ;
  inner CV chọn LogisticRegression/SVM/LightGBM. Tách riêng bảng lịch sử bảy classifier cũ và Dummy đối chứng.
- **Compact Early:** Kinematic nguyên chiều; giảm chiều riêng từng nhánh, rồi nối và chọn một trong
  ba classifier trên. MOMENT ANOVA-64 hoặc PCA-99%; GAF channel_concat (và adapter đối chứng)
  ANOVA-64/256 hoặc PCA-95%; CWT embedding gốc, none hoặc PCA-99% (none được phép thắng).
- **Late:** classifier riêng: Kinematic none/LightGBM cố định; MOMENT ANOVA-64/LR hoặc PCA-99%/SVM;
  GAF ANOVA-256/LightGBM, ANOVA-64/SVM hoặc PCA-95%/LR; CWT none/LR hoặc PCA-99%/LR.
  Chọn equal pooling hoặc trọng số simplex không âm tổng bằng 1 học bằng log-loss trên cross-fitted
  Train probabilities, regularization về uniform λ=0 hoặc 0.1. `equal__*` là đối chứng chỉ cho uniform.
- **Adaptive/Selective:** giữ Kinematic làm mô hình chính. Với |pK−0.5|≤τ (τ=0.1/0.25/0.5),
  p=pK+α×(Σw_aux p_aux−pK), α=0.25/0.5, simplex w_aux học từ cross-fitted probabilities,
  λ=0.1. Ngoài vùng bất định, giữ nguyên pK. Inner CV chỉ chấp nhận ứng viên nếu mean F1 tăng
  nghiêm ngặt **và** mean Accuracy không giảm; nếu không, trả về đúng Kinematic. Đây là quy tắc chọn
  trên Train, **không bảo đảm** không suy giảm trên outer Test. Gate không giảm chi phí encoder
  trong thí nghiệm vì mọi embedding đã cache và mọi nhánh đều được dự đoán.

Lưới có **287 ứng viên**, gồm cả đơn nhánh và fallback; `34` quy trình nested,
6 outer jobs, 78 stage fit (6×[3 inner chính + 9 cross-fit inner Train + 1 outer]).
`settings.json` giữ đầy đủ classifier params, grid, seed, versions và SHA-256 code/input.
Lưới nhỏ dựa trên nghiên cứu trước, không quét mọi reducer/classifier đã có.

## Chống data leakage và checkpoint

Mỗi branch bank median imputer → variance filter → StandardScaler chỉ fit trên Train của split.
PCA full SVD và ANOVA dùng đúng Train đó; Kinematic không giảm chiều. PCA/selector không được
chép từ nghiên cứu trước hoặc từ outer Train xuống inner Train. Chia sẻ tính toán chỉ giữa các
ứng viên có cùng tập Train. PCA không whitening và không rescale component sau projection.

Để đánh giá một cấu hình Late/Adaptive trên inner Validation, trước hết cross-fit **3-fold riêng
trong inner Train** cho từng base model, refit cả preprocessing/reducer. Bộ kết hợp chỉ học từ
các dự đoán này và nhãn inner Train, sau đó nhận xác suất Validation của base model fit đủ inner Train.
Vì vậy nhãn của một inner Validation không đi vào preprocessing, base model hay combiner của split đó.
Sau khi chọn bằng ba inner Validation, outer combiner fit từ cross-fitted base predictions của
215 outer Train (ba inner chính); base models refit đủ 215 người rồi dự đoán 43 outer Test.

Validator dựng lại split/coverage, kiểm tra fit IDs của từng bank và combiner, tái tính toàn bộ
inner scores/selection từ predictions, refit combiner từ đúng cross-fit cache, nạp và chạy lại
outer checkpoints, tái tính metrics từ OOF. Không dùng mô hình train full cohort làm bằng chứng OOF.
Lưu 6 checkpoint base/bank/selected Early và 6 checkpoint combiner trong `checkpoints/`.
Các joblib ở máy (gitignored để tránh repository quá lớn); script tái tạo dùng cache có sẵn.

## Bảng so sánh chính

Đơn vị **%**, Δ là **điểm phần trăm**. OOF gộp và mean±SD sáu fold được tách cột;
SD giữa các fold có tập Train chồng lấp **không phải CI**. Bảng đầy đủ ở `nested_summary.csv`.
`mean_raw_features` là chiều cache của các nhánh được chọn; `mean_features` là tổng chiều thực
sự vào classifier sau preprocessing/reducer. Kinematic vẫn dùng đủ 1.036 đặc trưng gốc,
chỉ loại cột hằng theo Train như baseline (trung bình còn 1.027,67 cột), không PCA/feature selection.
Nếu Adaptive fallback về Kinematic, số chiều của hàng đó chỉ tính mô hình đang đóng góp;
chi phí thực nghiệm vẫn bao gồm tính xác suất các nhánh phụ để chọn cấu hình.

| scenario | mean_raw_features | mean_features | OOF_accuracy | fold_accuracy mean±SD | OOF_f1_macro | fold_f1_macro mean±SD | OOF_auc | fold_auc mean±SD | OOF_balanced_accuracy | fold_balanced_accuracy mean±SD | Δ OOF F1 pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kinematic_LightGBM | 1036.00 | 1027.67 | 84.88 | 84.88 ± 3.21 | 84.72 | 84.70 ± 3.24 | 90.69 | 90.40 ± 3.13 | 84.75 | 84.73 ± 3.23 | 0.00 |
| original__KM | 15372.00 | 15363.67 | 75.58 | 75.58 ± 6.37 | 75.22 | 75.06 ± 6.75 | 83.48 | 83.62 ± 4.17 | 75.16 | 75.25 ± 6.84 | -9.49 |
| original__KG | 11788.00 | 11779.67 | 75.58 | 75.58 ± 5.04 | 75.22 | 75.04 ± 5.48 | 84.78 | 85.11 ± 5.31 | 75.16 | 75.19 ± 5.65 | -9.49 |
| original__KC | 26124.00 | 26115.67 | 79.46 | 79.46 ± 5.39 | 79.07 | 78.94 ± 5.78 | 86.40 | 86.03 ± 2.35 | 78.91 | 78.99 ± 5.85 | -5.64 |
| original__KMG | 26124.00 | 26115.67 | 74.81 | 74.81 ± 6.48 | 74.48 | 74.29 ± 6.63 | 83.90 | 84.56 ± 5.17 | 74.46 | 74.44 ± 6.71 | -10.23 |
| compact__KM | 15372.00 | 1161.17 | 79.84 | 79.84 ± 3.80 | 79.64 | 79.56 ± 4.06 | 88.41 | 88.57 ± 2.66 | 79.69 | 79.67 ± 4.31 | -5.08 |
| compact__KG | 76300.00 | 1186.33 | 79.07 | 79.07 ± 7.35 | 78.86 | 78.80 ± 7.23 | 87.41 | 87.81 ± 3.87 | 78.91 | 78.96 ± 6.90 | -5.86 |
| compact__KC | 26124.00 | 9525.83 | 83.33 | 83.33 ± 4.51 | 83.12 | 83.04 ± 4.79 | 89.44 | 89.20 ± 4.07 | 83.09 | 83.06 ± 4.88 | -1.60 |
| compact__KMG | 90636.00 | 1329.00 | 81.01 | 81.01 ± 5.18 | 80.76 | 80.71 ± 5.13 | 88.40 | 89.71 ± 2.67 | 80.74 | 80.81 ± 4.77 | -3.95 |
| late__KM | 15372.00 | 1161.17 | 83.33 | 83.33 ± 6.14 | 83.15 | 83.16 ± 6.15 | 90.09 | 90.44 ± 1.93 | 83.18 | 83.18 ± 6.05 | -1.57 |
| late__KG | 76300.00 | 1197.50 | 81.01 | 81.01 ± 3.42 | 80.73 | 80.72 ± 3.36 | 91.00 | 91.10 ± 2.83 | 80.65 | 80.64 ± 3.31 | -3.99 |
| late__KC | 26124.00 | 17820.67 | 85.27 | 85.27 ± 3.50 | 85.10 | 85.09 ± 3.52 | 91.80 | 91.72 ± 2.37 | 85.10 | 85.13 ± 3.49 | 0.38 |
| late__KMG | 90636.00 | 1354.17 | 82.95 | 82.95 ± 4.80 | 82.68 | 82.67 ± 4.84 | 90.46 | 90.15 ± 2.69 | 82.57 | 82.60 ± 4.85 | -2.04 |
| adaptive__KM | 15372.00 | 1161.17 | 83.72 | 83.72 ± 6.06 | 83.55 | 83.56 ± 6.08 | 89.95 | 90.44 ± 1.91 | 83.61 | 83.59 ± 5.98 | -1.16 |
| adaptive__KG | 76300.00 | 1218.33 | 83.33 | 83.33 ± 4.01 | 83.12 | 83.10 ± 4.06 | 90.82 | 90.69 ± 1.78 | 83.09 | 83.09 ± 4.12 | -1.60 |
| adaptive__KC | 26124.00 | 5378.50 | 85.66 | 85.66 ± 4.01 | 85.56 | 85.55 ± 3.97 | 91.16 | 91.09 ± 3.06 | 85.79 | 85.80 ± 3.63 | 0.85 |
| adaptive__KMG | 90636.00 | 1340.83 | 82.95 | 82.95 ± 5.23 | 82.77 | 82.76 ± 5.20 | 89.90 | 90.36 ± 1.14 | 82.83 | 82.85 ± 5.01 | -1.95 |
| all_selected | 63158.67 | 1243.83 | 81.78 | 81.78 ± 5.78 | 81.52 | 81.44 ± 5.99 | 88.65 | 89.05 ± 2.79 | 81.44 | 81.40 ± 6.05 | -3.20 |

## Đối chứng GAF chi phí thấp

| scenario | mean_features | accuracy | f1_macro | auc | balanced_accuracy |
| --- | --- | --- | --- | --- | --- |
| compact__KG_adapter | 1130.33 | 77.13 | 76.88 | 86.77 | 76.90 |
| compact__KMG_adapter | 1345.00 | 79.46 | 79.16 | 88.01 | 79.08 |
| late__KG_adapter | 1162.33 | 83.33 | 83.09 | 89.41 | 83.01 |
| equal__KG_adapter | 1123.67 | 83.72 | 83.32 | 88.48 | 83.02 |
| late__KMG_adapter | 1359.83 | 83.33 | 83.12 | 90.71 | 83.09 |
| equal__KMG_adapter | 1303.50 | 78.68 | 78.45 | 88.51 | 78.47 |
| adaptive__KG_adapter | 1194.33 | 84.11 | 83.84 | 89.31 | 83.71 |
| adaptive__KMG_adapter | 1345.50 | 84.11 | 83.84 | 90.48 | 83.71 |

`ablation_comparison.csv` so sánh channel_concat với adapter trong cùng phương pháp, và
KMG với KM/KG. Đây là ablation **quy trình**: từng tổ hợp được chọn reducer/classifier/weights
riêng trong Train, không phải hiệu ứng nhân quả thuần của một tọa độ. Không gán mọi cải thiện
cho thêm một nhánh khi cấu hình học thay đổi.

## So với Early Fusion đã có

Bảng sau chỉ tóm tắt classifier lịch sử có mean F1 cao nhất mỗi tổ hợp (bảy classifier và Dummy đối chứng);
**đã chọn theo outer grid cũ**, có selection bias, không xem là nested estimate. Toàn bộ các hàng
lịch sử giữ ở `historical_early_summary.csv`. Original Early mới chỉ chọn trong ba classifier
đại diện bằng inner CV, nên khác pool và khác quy trình chọn. Baseline chính vẫn là LightGBM
Kinematic cố định, không đổi sang classifier do inner CV chọn.

| variant | classifier | accuracy | f1_macro | auc |
| --- | --- | --- | --- | --- |
| KG | XGBoost | 79.84 | 79.40 | 87.47 |
| KC | LightGBM | 79.46 | 79.07 | 86.40 |
| KMG | XGBoost | 77.13 | 76.70 | 84.97 |
| KM | XGBoost | 77.13 | 76.48 | 85.19 |

## Nhánh nào sửa đúng, nhánh nào gây hại

| scenario | corrected | harmed | same_correct | same_wrong |
| --- | --- | --- | --- | --- |
| Kinematic_LightGBM | 0 | 0 | 219 | 39 |
| original__KM | 12 | 36 | 183 | 27 |
| original__KG | 14 | 38 | 181 | 25 |
| original__KC | 14 | 28 | 191 | 25 |
| original__KMG | 13 | 39 | 180 | 26 |
| compact__KM | 6 | 19 | 200 | 33 |
| compact__KG | 13 | 28 | 191 | 26 |
| compact__KC | 12 | 16 | 203 | 27 |
| compact__KMG | 15 | 25 | 194 | 24 |
| late__KM | 5 | 9 | 210 | 34 |
| late__KG | 5 | 15 | 204 | 34 |
| late__KC | 9 | 8 | 211 | 30 |
| late__KMG | 7 | 12 | 207 | 32 |
| adaptive__KM | 6 | 9 | 210 | 33 |
| adaptive__KG | 6 | 10 | 209 | 33 |
| adaptive__KC | 8 | 6 | 213 | 31 |
| adaptive__KMG | 6 | 11 | 208 | 33 |
| all_selected | 5 | 13 | 206 | 34 |

`prediction_changes.csv` lưu từng người (baseline/fusion probabilities, corrected/harmed);
`confidence_ablation.csv` chia độ tự tin Kinematic thành năm khoảng định trước. Những trường hợp
này là mô tả hậu nghiệm, không dùng để huấn luyện gate hoặc chỉnh threshold sau khi xem Test.
`selected_by_fold.csv` ghi fallback và tham số combiner; `selection_frequency.csv` mô tả độ ổn định.
Xem dấu và độ lớn chênh lệch từng fold trong notebook trước khi quyết định giữ một nhánh.

### Phương pháp đáng giữ và giới hạn của mức tăng

Hàng mean F1 cao nhất mỗi phương pháp, **xếp hạng mô tả sau outer CV**, không dùng các hàng này
để chọn lại tham số hoặc tạo OOF mới:

| method | scenario | OOF_Accuracy | OOF_F1 | mean_F1 | OOF_AUC | delta_F1_pp | corrected | harmed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | original__KC | 79.46 | 79.07 | 78.94 | 86.40 | -5.64 | 14 | 28 |
| compact | compact__KC | 83.33 | 83.12 | 83.04 | 89.44 | -1.60 | 12 | 16 |
| late | late__KC | 85.27 | 85.10 | 85.09 | 91.80 | 0.38 | 9 | 8 |
| adaptive | adaptive__KC | 85.66 | 85.56 | 85.55 | 91.16 | 0.85 | 8 | 6 |

Quy trình chọn tổng thể `all_selected`: Accuracy OOF **81.78%**
(-3.10 điểm), F1 OOF **81.52%**
(-3.20 điểm), mean F1 **81.44%**
so với 84.70% của Kinematic. Giữ Kinematic + LightGBM làm mô hình chính. Quy trình mở rộng chọn Fusion chưa đồng thời tăng mean F1 và giữ Accuracy OOF; không thay baseline bằng một hàng thắng hậu nghiệm.
Một người được dự đoán đúng thêm tương ứng khoảng 0.388 điểm Accuracy trên 258 người;
không xem vài trường hợp sửa đúng là bằng chứng chắc chắn của tổng quát hóa.

Tác động thêm từng nhánh (điểm phần trăm OOF):

| method | effect | delta_oof_accuracy_pp | delta_oof_f1_macro_pp | delta_oof_auc_pp |
| --- | --- | --- | --- | --- |
| original | M | -9.30 | -9.49 | -7.21 |
| original | G | -9.30 | -9.49 | -5.91 |
| original | C | -5.43 | -5.64 | -4.29 |
| original | G given KM | -0.78 | -0.74 | 0.42 |
| original | M given KG | -0.78 | -0.74 | -0.88 |
| compact | M | -5.04 | -5.08 | -2.28 |
| compact | G | -5.81 | -5.86 | -3.28 |
| compact | C | -1.55 | -1.60 | -1.25 |
| compact | G given KM | 1.16 | 1.13 | -0.01 |
| compact | M given KG | 1.94 | 1.91 | 0.99 |
| late | M | -1.55 | -1.57 | -0.60 |
| late | G | -3.88 | -3.99 | 0.31 |
| late | C | 0.39 | 0.38 | 1.11 |
| late | G given KM | -0.39 | -0.47 | 0.36 |
| late | M given KG | 1.94 | 1.95 | -0.54 |
| adaptive | M | -1.16 | -1.16 | -0.74 |
| adaptive | G | -1.55 | -1.60 | 0.13 |
| adaptive | C | 0.78 | 0.85 | 0.47 |
| adaptive | G given KM | -0.78 | -0.78 | -0.05 |
| adaptive | M given KG | -0.39 | -0.35 | -0.92 |

Các nhánh phụ có thể bổ sung khi lỗi của chúng khác Kinematic, nhưng cũng có thể kéo xác suất
của một dự đoán đúng qua ngưỡng 0.5. Bảng corrected/harmed đo trực tiếp hai chiều tác động này.
Chênh lệch GAF concat/adapter cần cân nhắc chi phí encode ~7 lần; CWT không được ưu tiên chỉ
vì có một fold tốt. Fallback nhiều lần cho biết inner CV không tìm thấy lợi ích theo safety rule,
không chứng minh nhánh đó hoàn toàn không chứa thông tin. Không suy nguyên nhân sinh học từ ablation.

## Chi phí và ý nghĩa thực tế

Tổng thời gian các thành phần fit/transform đã đo: **38.56 phút** (tổng workload, có
chạy hai outer jobs đồng thời; **không phải wall time**). Checkpoint nén **640.3 MiB**.
`compute_costs.csv` tách stage/bank/classifier và thời gian predict, không bao gồm mọi thao tác I/O,
meta fitting hoặc trích xuất encoder. Late/Adaptive dùng thêm cross-fitting nên chi phí train lớn hơn
một Early cố định; tổng thời gian tìm kiếm không đồng nghĩa latency triển khai.
GAF channel_concat vẫn cần bảy lần encode kênh; benchmark trước khoảng 7.21× embedding gốc,
adapter khoảng 0.99×. Giảm chiều chỉ giảm classifier input, không loại bỏ chi phí encoder.
Không có benchmark end-to-end Fusion trên tín hiệu thô trong nhiệm vụ này.

Ưu tiên chọn phương pháp đơn giản nếu chỉ tăng rất ít. Quy trình `all_selected` cho phép trả về
Kinematic qua safety rule, nhằm kiểm tra liệu việc mở pool Fusion có lợi khi lựa chọn hoàn toàn
trong Train. Không dùng hàng thắng outer để triển khai tự động hoặc ghi đè baseline.

Thiết kế đã tham khảo kết quả trên **cùng cohort 258 người**, nên nested CV đợt này không phải một
tập xác nhận mới độc lập. Không suy ra ý nghĩa thống kê hay hiệu quả chẩn đoán lâm sàng từ sáu fold;
cần external holdout hoặc cohort khác để xác nhận mức tăng nhỏ. Không thực hiện XAI.

## Artifacts và chạy lại

Giữ nguyên 1016 artifact cũ bằng SHA-256. Kết quả mới chỉ ở `results/fusion_ablation/`;
notebook mới `notebooks/08_fusion_ablation.ipynb`, tests `tests/test_fusion.py`.
`stages/` lưu cross-fit probabilities, fit IDs, selected features, PCA variance và chi phí;
`selections/` lưu toàn bộ inner candidates/scores/predictions; `folds/`, `oof/`,
`fold_scores.csv`, `nested_oof.csv`, PNG/SVG và manifest phục vụ kiểm chứng.
`code/` giữ bản sao mã thực nghiệm và phân tích để review độc lập.

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/evaluate_fusion_ablation.py --workers 2
.venv/Scripts/python.exe -X utf8 -B -u scripts/report_fusion_ablation.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_fusion_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 08_fusion_ablation.ipynb --validation-output results/fusion_ablation/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_fusion_ablation.py
```

Evaluation có checkpoint và từ chối trộn input/code/protocol đã đổi. Tạo lại báo cáo/notebook
không thay OOF. Cơ sở kỹ thuật: [StackingClassifier — cross-validated training of final estimator](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.StackingClassifier.html),
[cross_val_predict](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.cross_val_predict.html),
[nested CV](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
