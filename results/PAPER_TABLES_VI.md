# Đối chiếu 15 cấu hình trong các bảng của bài báo

Đã hoàn tất cả 15 cấu hình được công bố ở Bảng I, II và III. Đây là bản dựng thực nghiệm độc lập từ PDF và dữ liệu hiện có; không có mã nguồn, phụ lục hoặc cấu hình gốc của tác giả để xác nhận tái lập chính xác.

Kết quả chính: động học + LightGBM đạt accuracy **84,88%**. Fusion động học + MOMENT + GAF với AdaBoost đạt **73,26%**, thấp hơn mức **85,3%** công bố **12,04 điểm phần trăm**. Vì vậy, **chưa tái hiện được kết quả fusion trọng tâm của bài báo**.

## Cách đọc các bảng

Mỗi ô kết quả ghi **Accuracy / F1 / AUC**, đơn vị %. Cột thực nghiệm dùng trung bình của 6 fold, F1 là **macro-F1**. Bài báo không nêu rõ cách lấy trung bình F1 nên hai giá trị F1 chỉ đối chiếu tham khảo. AUC trung bình các fold có thể khác AUC tính gộp toàn bộ dự đoán OOF; AUC gộp được giữ riêng trong CSV. Δ Accuracy = thực nghiệm − bài báo, đơn vị điểm phần trăm.

Đánh giá trên 258 người (115 nhãn 0, 143 nhãn 1), chia stratified 6-fold theo người, seed 42. Mỗi fold kiểm tra có 43 người. Các bước chuẩn hóa và bộ phân loại chỉ được fit trên tập huấn luyện của fold tương ứng. Fusion dùng GAF/ViT-B16 và CWT/EfficientNet-B4 đã xác định trong quy trình.

## Bảng I — MOMENT và đặc trưng động học

| Cấu hình | Bài báo: Acc / F1 / AUC | Thực nghiệm: Acc / F1 macro / AUC | Δ Accuracy |
|---|---:|---:|---:|
| MOMENT + Logistic Regression | 65.1 / 65.1 / 70.1 | 63.18 / 62.12 / 69.96 | -1.92 |
| MOMENT + Random Forest | 65.1 / 63.3 / 71.4 | 67.83 / 64.48 / 72.64 | +2.73 |
| MOMENT + SVM | 70.2 / 69.9 / 74.9 | 67.83 / 64.79 / 75.07 | -2.37 |
| Động học + XGBoost | 82.2 / 81.9 / 89.9 | 84.11 / 83.77 / 91.13 | +1.91 |
| Động học + LightGBM | 82.9 / 82.8 / 90.5 | 84.88 / 84.70 / 90.40 | +1.98 |
| Động học + AdaBoost | 83.7 / 83.6 / 90.1 | 81.01 / 80.76 / 88.94 | -2.69 |

## Bảng II — Biểu diễn ảnh

| Cấu hình | Bài báo: Acc / F1 / AUC | Thực nghiệm: Acc / F1 macro / AUC | Δ Accuracy |
|---|---:|---:|---:|
| GAF / EfficientNet-B4 + XGBoost | 65.9 / 65.3 / 71.8 | 66.28 / 64.25 / 74.34 | +0.38 |
| GAF / VGG16 + Random Forest | 68.6 / 67.0 / 72.2 | 68.22 / 65.97 / 71.41 | -0.38 |
| GAF / ViT-B16 + LightGBM | 73.3 / 73.0 / 78.5 | 63.18 / 61.65 / 66.30 | -10.12 |
| CWT / VGG16 + SVM | 66.3 / 66.0 / 70.8 | 57.75 / 41.42 / 58.40 | -8.55 |
| CWT / ViT-B16 + SVM | 67.1 / 66.4 / 72.1 | 68.60 / 66.89 / 72.56 | +1.50 |
| CWT / EfficientNet-B4 + LightGBM | 71.7 / 71.5 / 75.3 | 63.57 / 62.50 / 69.82 | -8.13 |

## Bảng III — Fusion (AdaBoost)

| Cấu hình | Bài báo: Acc / F1 / AUC | Thực nghiệm: Acc / F1 macro / AUC | Δ Accuracy |
|---|---:|---:|---:|
| Động học + MOMENT + GAF + CWT | 83.7 / 83.7 / 90.1 | 75.19 / 74.63 / 82.77 | -8.51 |
| Động học + GAF | 84.9 / 84.9 / 91.1 | 77.91 / 77.30 / 86.38 | -6.99 |
| Động học + MOMENT + GAF | 85.3 / 85.2 / 90.8 | 73.26 / 72.63 / 82.62 | -12.04 |

## Giới hạn ảnh hưởng đến kết luận

- Nhãn chính là trường `group`. Có 5 người mà nhãn này khác cờ chẩn đoán dysgraphia và 3 người có cờ đề nghị loại. Các trường hợp này và phân tích độ nhạy đã được ghi trong báo cáo đầy đủ; không đổi cohort sau khi xem kết quả.
- Bài báo chưa cung cấp công thức đầy đủ của 74 đặc trưng. Các công thức hiện dùng được dựng lại theo số lượng và nhóm đặc trưng đã mô tả.
- Checkpoint, cách chuyển encoder sang 7 kênh, chi tiết CWT và siêu tham số bộ phân loại chưa được mô tả đủ để khôi phục duy nhất. Các lựa chọn hiện dùng, gồm AdaBoost SAMME, được ghi rõ trong quy trình.
- Cùng số lượng 258 người không xác nhận cùng danh sách người và fold của tác giả. Các chênh lệch này là khả năng giải thích cần kiểm chứng, chưa phải nguyên nhân đã được chứng minh của khoảng cách kết quả.

## Tệp để kiểm tra và sử dụng

- [CSV của đúng 15 cấu hình, theo thứ tự bảng](paper_tables_comparison.csv).
- [Dự đoán OOF và kết quả từng cấu hình](cv/) và [kiểm tra tính đầy đủ](validation.json).
- [Báo cáo đầy đủ, biểu đồ và phân tích độ nhạy](REPORT_VI.md).
- [Quy trình và toàn bộ giả định triển khai](../EXPERIMENT_PROTOCOL.md).
- [Notebook đánh giá đã chạy](../notebooks/04_evaluation_and_fusion.ipynb).

Nguồn giá trị công bố: PDF người dùng cung cấp, *Towards Objective Dysgraphia Detection: A Multi-Branch Deep Learning Approach for Online Handwriting Analysis*, Bảng I–III ([bản arXiv](https://arxiv.org/abs/2607.09826v1)). Giá trị thực nghiệm được lấy từ `comparison_paper.csv`; không huấn luyện lại hay chọn lại tham số khi tạo báo cáo này.

Tạo lại phần đối chiếu sau khi đánh giá: `.venv/Scripts/python.exe scripts/report_paper_tables.py`.
