# DiaGraMo: tái lập thực nghiệm đa nhánh

Tái lập độc lập bài *Towards Objective Dysgraphia Detection: A Multi-Branch Deep Learning Approach for Online Handwriting Analysis* (Ouhib và cộng sự, 2026).

- [Đối chiếu ưu tiên: 15 cấu hình ở Bảng I–III của bài báo](results/PAPER_TABLES_VI.md)
- [Báo cáo và kết quả đã chạy](results/REPORT_VI.md)
- [Quy trình, giả định và giới hạn tái lập](EXPERIMENT_PROTOCOL.md)
- [Bảng kết quả đầy đủ](results/cv/summary.csv)
- [Đối chiếu với bài báo](results/comparison_paper.csv)
- [Kiểm tra tiến độ/tính đầy đủ](results/validation.json)

Không diễn giải một pipeline đã viết hoặc một lượt chạy đang dở là thực nghiệm đã hoàn tất. `results/validation.json` chỉ có `complete=true` sau khi đủ 168 kết quả CV: 21 cấu hình × 7 bộ phân loại + 1 Dummy. Mọi kết quả có tệp dự đoán OOF tương ứng.

## Thiết lập trên Windows

Python 3.10 được dùng trong `.venv` của dự án. Môi trường `.venv314` ban đầu được giữ nguyên. Dữ liệu nguồn không bị sửa.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

`requirements-lock.txt` là bản chụp môi trường đã chạy, bao gồm phụ thuộc gián tiếp. CUDA không cần thiết cho bước phân loại; có thể chạy encoder bằng `--device cpu`, nhưng chậm hơn. T1000 4 GB đã được đo và chọn float32 thay vì float16; không mặc định rằng float16 luôn nhanh hơn.

## Chạy lại

Từ thư mục gốc dự án:

```powershell
.venv/Scripts/python.exe scripts/prepare.py
.venv/Scripts/python.exe scripts/download_weights.py
.venv/Scripts/python.exe scripts/run_all.py --skip-download
.venv/Scripts/python.exe scripts/sensitivity.py
.venv/Scripts/python.exe scripts/report.py
.venv/Scripts/python.exe scripts/report_paper_tables.py
.venv/Scripts/python.exe scripts/execute_notebooks.py
.venv/Scripts/python.exe scripts/train_selected.py
```

Hoặc chạy riêng một nhánh:

```powershell
.venv/Scripts/python.exe scripts/extract.py --representation moment --batch-size 8
.venv/Scripts/python.exe scripts/extract.py --representation gaf --encoder vit_b_16 --batch-size 16
.venv/Scripts/python.exe scripts/extract.py --representation cwt --encoder efficientnet_b4 --batch-size 16
.venv/Scripts/python.exe scripts/evaluate.py --only kinematic
```

Các tên encoder còn lại: `resnet50`, `efficientnet_b4`, `vit_b_16`, `vgg16`. Trích xuất có checkpoint theo tác vụ; chạy lại cùng lệnh sẽ tiếp tục. Tệp `.npz` hoàn chỉnh được bỏ qua. Các kết quả CV JSON đã có cũng được bỏ qua. Khi thay công thức đặc trưng, checkpoint hoặc tham số, cần dùng một thư mục kết quả mới hoặc di chuyển cache sinh ra trước đó; không trộn kết quả từ các thiết lập khác nhau.

`train_selected.py` chỉ chạy sau khi toàn bộ CV đã được kiểm tra. Script fit lại cấu hình đứng đầu CV và cấu hình ba nhánh/AdaBoost của bài báo trên toàn cohort. Các mô hình này phục vụ nạp đặc trưng để suy luận; đánh giá hiệu năng vẫn phải dùng dự đoán OOF đã lưu.

`scripts/run_all.py` chạy encoder nối tiếp trên GPU và tận dụng CPU đánh giá đặc trưng đã xong. Log ở `results/logs/`; tiến độ hiện hành tại `results/logs/run_status.json` và `data/derived/*.progress.json`. Chỉ tải các trọng số công khai; không gửi dữ liệu người tham gia ra ngoài.

## Notebook

1. `notebooks/01_data_inspection.ipynb`: kiểm tra cohort, dữ liệu thiếu, timestamp và nhãn.
2. `notebooks/02_kinematic_features.ipynb`: schema 74 đặc trưng và ma trận 258×1.036.
3. `notebooks/03_pretrained_representations.ipynb`: các embedding và minh họa GAF/CWT.
4. `notebooks/04_evaluation_and_fusion.ipynb`: fold, kết quả CV, đối chiếu và biểu đồ.

Chọn kernel Python của `.venv`. Notebook đọc các artifact đã sinh, không tự khởi động một lượt GPU dài khi mở.

## Dữ liệu và cách đánh giá

258 người × 14 tác vụ × 7 kênh; nhãn chính `group` (115/143). Không dùng điểm nhận thức làm đầu vào. Giữ thứ tự ID/tác vụ cố định. Đánh giá StratifiedKFold 6 fold, seed 42, tách theo người. Scaler và bộ lọc cột hằng chỉ fit trong train. Báo cáo accuracy, cả ba biến thể F1, ROC-AUC, balanced accuracy, OOF và mean±SD của 6 fold.

10 cấu hình đơn nhánh và 11 fusion. Fusion cố định GAF=ViT-B/16, CWT=EfficientNet-B4 theo bài báo. Bảy classifier: Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, AdaBoost, MLP; Dummy là kiểm tra mức nền bổ sung.

Định nghĩa nhãn, danh sách ID chính thức, công thức từng đặc trưng, scale CWT, chi tiết encoder và siêu tham số chưa được bài báo công bố đầy đủ. Tất cả lựa chọn tái lập được ghi trong protocol; trùng 258 người hoặc đạt một tỷ lệ tương tự không đủ xác nhận tái hiện chính xác kết quả tác giả.

## Kiểm tra kết quả

```powershell
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe scripts/report.py --no-notebooks
```

`scripts/report.py` kiểm tra đủ cấu hình, duy nhất ID và tính hợp lệ của mọi dự đoán OOF trước khi báo hoàn tất. Cờ `--no-notebooks` giữ nguyên các notebook đã thực thi khi tạo lại báo cáo.
