# DiaGraMo: tái lập thực nghiệm đa nhánh

## Tối ưu đặc trưng

Thực nghiệm PCA, SelectKBest ANOVA và SelectFromModel ExtraTrees nằm ngay trong project:
[báo cáo so sánh](results/feature_optimization/REPORT_VI.md),
[bảng đầy đủ](results/feature_optimization/summary.csv) và
[đánh giá chọn cấu hình bằng nested CV](results/feature_optimization/nested_comparison.csv).
Kết quả cũ được giữ nguyên; kết quả mới lưu riêng tại `results/feature_optimization/`.

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/optimize_features.py --workers 2
.venv/Scripts/python.exe -X utf8 -B scripts/optimize_features.py --report-only
.venv/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -v
```

Giữ nguyên 6 fold theo participant, seed 42. Imputer, scaler, PCA và selector chỉ fit
trong train của từng split; lựa chọn phương pháp/mức chiều/classifier dùng inner CV
riêng trong outer train. `validation.json` của thư mục mới phải có `complete=true`
trước khi xem đợt chạy là hoàn tất. Khi đổi code/tham số/đặc trưng, dùng `--output`
trỏ tới một thư mục kết quả mới để không trộn các đợt chạy.

## Embedding Extraction và Ablation

Thử nghiệm mới giữ frozen GAF/ViT-B16 và CWT/EfficientNet-B4, so sánh embedding gốc
với adapter cố định 7→3, embedding trung bình sau khi encode từng kênh, và embedding
nối riêng bảy kênh. Kết hợp PCA/ANOVA/ExtraTrees bên trong train của mỗi split.
Không triển khai Fusion. Baseline, checkpoint và kết quả trước được giữ nguyên.

- [Báo cáo tiếng Việt](results/embedding_ablation/REPORT_VI.md)
- [Bảng số 192 cấu hình](results/embedding_ablation/summary.csv)
- [Ablation cùng classifier/reducer](results/embedding_ablation/ablation_matched.csv)
- [Nested CV](results/embedding_ablation/nested_summary.csv)
- [Notebook 07: bảng số và biểu đồ](notebooks/07_embedding_ablation.ipynb)
- [Validation](results/embedding_ablation/validation.json): chỉ `complete=true` khi
  số liệu, notebook, PNG/SVG và kiểm thử đã thành công.

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

Ma trận embedding `.npy` lớn nằm tại `results/embedding_ablation/embeddings/` trên máy,
được bỏ qua trong Git; metadata, SHA-256, OOF, bảng và notebook vẫn được lưu trong repo.
Script trích xuất có thể tạo lại các ma trận từ cùng dữ liệu/checkpoint cục bộ.

## Thực nghiệm baseline

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
5. [05_feature_optimization.ipynb](notebooks/05_feature_optimization.ipynb): bảng baseline, biểu đồ phương pháp, heatmap 216 cấu hình, đường cong số chiều và mức nén.
6. [06_nested_cv_and_stability.ipynb](notebooks/06_nested_cv_and_stability.ipynb): nested CV, chênh lệch từng fold, ROC/OOF, confusion matrix, độ ổn định selector và phương sai PCA trên train.

Hai notebook mới đã có output; biểu đồ PNG/SVG và manifest nguồn ở
`results/feature_optimization/figures/`. Chạy riêng chúng, giữ nguyên output notebook cũ:

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 05_feature_optimization.ipynb 06_nested_cv_and_stability.ipynb --validation-output results/feature_optimization/notebook_validation.json
```

Nếu cần tạo lại nội dung hai notebook mới trước khi chạy, dùng
`.venv/Scripts/python.exe -X utf8 -B scripts/create_feature_notebooks.py`.

Chọn kernel Python của `.venv`. Notebook đọc các artifact đã sinh, không tự khởi động một lượt GPU dài khi mở.

## Dữ liệu và cách đánh giá

258 người × 14 tác vụ × 7 kênh; nhãn chính `group` (115/143). Không dùng điểm nhận thức làm đầu vào. Giữ thứ tự ID/tác vụ cố định. Đánh giá StratifiedKFold 6 fold, seed 42, tách theo người. Scaler và bộ lọc cột hằng chỉ fit trong train. Báo cáo accuracy, cả ba biến thể F1, ROC-AUC, balanced accuracy, OOF và mean±SD của 6 fold.

10 cấu hình đơn nhánh và 11 fusion. Fusion cố định GAF=ViT-B/16, CWT=EfficientNet-B4 theo bài báo. Bảy classifier: Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, AdaBoost, MLP; Dummy là kiểm tra mức nền bổ sung.

Định nghĩa nhãn, danh sách ID chính thức, công thức từng đặc trưng, scale CWT, chi tiết encoder và siêu tham số chưa được bài báo công bố đầy đủ. Tất cả lựa chọn tái lập được ghi trong protocol; trùng 258 người hoặc đạt một tỷ lệ tương tự không đủ xác nhận tái hiện chính xác kết quả tác giả.

## Kiểm tra kết quả

### Explainable AI trên outer Test

Kết quả XAI tại [results/xai/REPORT_VI.md](results/xai/REPORT_VI.md) và
[notebooks/09_xai.ipynb](notebooks/09_xai.ipynb): TreeSHAP cho Kinematic + LightGBM,
mapping 1.036 feature/14 task, grouped permutation held-out và replay cơ chế Late/Adaptive KC.
Notebook có bảng số trước biểu đồ, đọc artifact đã kiểm chứng, không fit/chọn lại model.
Attribution raw được kiểm tra ở log-odds, probability SHAP được kiểm tra riêng; không coi
confidence gate là attention network hoặc gán PCA embedding cho tín hiệu sinh học.

```powershell
.venv/Scripts/python.exe -m pip install --no-deps -r requirements-xai.txt
.venv/Scripts/python.exe -X utf8 -B -u scripts/analyze_xai.py
.venv/Scripts/python.exe -X utf8 -B -u scripts/report_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/plot_xai.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_xai_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 09_xai.ipynb --validation-output results/xai/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_xai.py
```

Giữ nguyên checkpoint/OOF cũ; nếu checkpoint cần thiết thiếu, chỉ tái tạo đúng cấu hình đã chọn,
lưu ở `results/xai/reconstructed/` rồi đối chiếu OOF trước khi giải thích. Không thay baseline
hoặc tối ưu Accuracy bằng XAI. Chỉ xem hoàn tất khi `results/xai/validation.json` có `complete: true`.

### Multimodal Fusion với nested CV

Kết quả Fusion mới được lưu riêng tại `results/fusion_ablation/`, không thay baseline cũ.
So sánh Original Early, Compact Early, Late và Adaptive/Selective Fusion; giữ Kinematic
nguyên chiều + LightGBM làm baseline. Late/Adaptive học bộ kết hợp từ xác suất cross-fitted;
inner Train có cross-fitting 3-fold riêng để tránh nhãn inner Validation đi vào bộ kết hợp.
Mọi reducer và lựa chọn cấu hình chỉ dùng Train. Không trích xuất lại cache hoặc thực hiện XAI.

```powershell
.venv/Scripts/python.exe -X utf8 -B -u scripts/evaluate_fusion_ablation.py --workers 2
.venv/Scripts/python.exe -X utf8 -B -u scripts/report_fusion_ablation.py
.venv/Scripts/python.exe -X utf8 -B scripts/create_fusion_notebook.py
.venv/Scripts/python.exe -X utf8 -B scripts/execute_notebooks.py --only 08_fusion_ablation.ipynb --validation-output results/fusion_ablation/notebook_validation.json
.venv/Scripts/python.exe -X utf8 -B scripts/validate_fusion_ablation.py
```

[Notebook 08](notebooks/08_fusion_ablation.ipynb) có bảng số OOF, mean±SD theo fold,
ablation và PNG/SVG. [Báo cáo tiếng Việt](results/fusion_ablation/REPORT_VI.md) giải thích
kết quả, chi phí và giới hạn. Chỉ xem hoàn tất khi `results/fusion_ablation/validation.json`
có `complete: true`. Checkpoint joblib được lưu ở máy trong `results/fusion_ablation/checkpoints/`
và gitignored; các script, OOF, audit, bảng số và hình có thể đưa lên GitHub.

```powershell
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe scripts/report.py --no-notebooks
```

`scripts/report.py` kiểm tra đủ cấu hình, duy nhất ID và tính hợp lệ của mọi dự đoán OOF trước khi báo hoàn tất. Cờ `--no-notebooks` giữ nguyên các notebook đã thực thi khi tạo lại báo cáo.
