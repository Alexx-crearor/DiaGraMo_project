# DiaGraMo — tái lập thực nghiệm bốn nhánh

Tái lập độc lập bài *Towards Objective Dysgraphia Detection: A Multi-Branch Deep Learning Approach for Online Handwriting Analysis* (Ouhib và cộng sự, 2026). Đây là bản dựng từ mô tả bài báo và dữ liệu hiện có. Các lựa chọn triển khai được ghi trong [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Bốn nhánh biểu diễn

| Nhánh | Module trong `src/diagramo/branches/` | Biểu diễn baseline |
|---|---|---|
| Kinematic | `kinematic.py` | 74 đặc trưng/tác vụ, 1.036/người |
| MOMENT | `moment.py` | MOMENT-1-large đóng băng, 1.024/tác vụ, 14.336/người |
| GAF | `gaf.py` | Gramian Angular Summation Field, 7×224×224 |
| CWT | `cwt.py` | Morlet, 224 scale, ảnh 7×224×224 |

Công thức dùng chung vẫn ở `src/diagramo/features.py`; các module nhánh gọi trực tiếp cùng triển khai. Encoder ảnh nằm trong `scripts/extract.py`: ResNet50, EfficientNet-B4, ViT-B/16 và VGG16, dùng trọng số torchvision ImageNet đóng băng. Convolution đầu chuyển từ RGB sang bảy kênh bằng tổng kernel RGB chia 7 rồi lặp bảy lần. Trọng số và SHA-256: [weights manifest](references/weights_manifest.json).

## Dữ liệu, phân loại và fusion

Nguồn cục bộ: `data/clean_data/*_merged.json`. Pipeline chọn 258 người có đủ 14 tác vụ và nhãn `group` (115/143), theo thứ tự ID và task cố định. Bảy tín hiệu: x, y, pressure, tilt, azimuth, time, pen_status. Không dùng điểm nhận thức làm đặc trưng; tệp nguồn được giữ nguyên.

Kinematic tính trên tín hiệu gốc; MOMENT nội suy về 512 điểm; GAF nội suy về 224 điểm; CWT dùng tín hiệu 512 điểm. Protocol mô tả timestamp, chuẩn hóa và các giả định tái lập.

Stratified 6-Fold CV theo participant, shuffle=True, seed 42. Imputer, bộ lọc cột hằng và StandardScaler chỉ fit trong train fold. Bảy classifier cố định: Logistic Regression, SVM RBF, Random Forest, XGBoost, LightGBM, AdaBoost và MLP; thêm Dummy theo prior của train.

Original Early Fusion nối trực tiếp vector nhánh trước cùng pipeline phân loại. GAF=ViT-B/16 và CWT=EfficientNet-B4 cố định theo bài báo. Giữ 10 cấu hình đơn nhánh và đủ 11 tổ hợp fusion (6 đôi, 4 bộ ba, 1 bộ bốn), tổng cộng 21×8=168 kết quả CV. Báo cáo OOF và mean±SD qua sáu fold; SD không phải CI.

## Cấu trúc repository

```text
src/diagramo/
  branches/              # Kinematic, MOMENT, GAF, CWT
  features.py            # công thức dùng chung, API cũ
  kernels.py             # RBF kernel cho SVM baseline
scripts/
  prepare.py             # audit, cohort, tín hiệu và Kinematic
  download_weights.py    # trọng số công khai và manifest
  extract.py             # encoder đóng băng, cache theo tác vụ
  evaluate.py            # classifier, preprocessing, CV, Early Fusion
  run_all.py             # điều phối baseline
  report.py              # kiểm tra OOF, bảng, biểu đồ, notebook
  report_paper_tables.py  # đối chiếu 15 cấu hình Bảng I–III
  sensitivity.py         # phân tích nhạy cảm baseline
  train_selected.py      # fit toàn cohort phục vụ suy luận, sau CV
  execute_notebooks.py    # thực thi notebook xem kết quả
  audit_cleanup.py        # SHA-256 và 168 kết quả bảo toàn
notebooks/               # bốn notebook gốc 01–04
references/              # metadata trọng số
results/
  cv/                    # summary, settings, folds, 168 JSON/OOF
  audit/                 # audit cohort/tín hiệu và checksum nguồn
  sensitivity/           # kết quả nhạy cảm baseline
  figures/               # PNG/SVG và provenance
  REPORT_VI.md
  PAPER_TABLES_VI.md
  validation.json
docs/cleanup/            # kiểm toán, manifest và bàn giao refactor
data/                    # dữ liệu/cache cục bộ, Git ignore
.cache/                  # pretrained weights cục bộ, Git ignore
```

## Thiết lập môi trường

Môi trường đã chạy dùng Python 3.10; dependency được khóa trong `requirements-lock.txt`. Từ thư mục gốc trên Windows:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip check
```

Encoder có thể chạy CPU bằng `--device cpu`. Baseline dùng float32; phép đo trên Quadro T1000 4 GB ghi trong protocol. Môi trường `.venv314` cũ được giữ lại.

## Chạy baseline từ đầu

Các lệnh sau dành cho một lượt tái lập chủ động. Chúng có thể tạo lại artifact, báo cáo, notebook và bản chụp dependency; dùng bản sao project nếu cần giữ nguyên toàn bộ artifact đã công bố.

```powershell
.venv/Scripts/python.exe scripts/prepare.py
.venv/Scripts/python.exe scripts/download_weights.py
.venv/Scripts/python.exe scripts/run_all.py --skip-download
.venv/Scripts/python.exe scripts/sensitivity.py
.venv/Scripts/python.exe scripts/report.py --no-notebooks
.venv/Scripts/python.exe scripts/report_paper_tables.py
.venv/Scripts/python.exe scripts/execute_notebooks.py
```

Chạy riêng biểu diễn và đánh giá:

```powershell
.venv/Scripts/python.exe scripts/extract.py --representation moment --batch-size 8
.venv/Scripts/python.exe scripts/extract.py --representation gaf --encoder vit_b_16 --batch-size 16
.venv/Scripts/python.exe scripts/extract.py --representation cwt --encoder efficientnet_b4 --batch-size 16
.venv/Scripts/python.exe scripts/evaluate.py --only kinematic
.venv/Scripts/python.exe scripts/evaluate.py
```

Đổi `--encoder` để chạy bốn backbone ảnh. Trích xuất có checkpoint theo tác vụ và bỏ qua `.npz` đã hoàn tất; evaluation bỏ qua JSON đã có. Khi chủ động thay công thức, trọng số hay tham số, cần tách cache và kết quả của lượt chạy tương ứng. `evaluate.py --output` cho phép dùng thư mục CV khác.

Sau CV có thể chạy `scripts/train_selected.py` để fit mô hình phục vụ suy luận trên toàn cohort. Hiệu năng vẫn đọc từ OOF.

## Xem và kiểm tra kết quả

- [Báo cáo baseline](results/REPORT_VI.md), [đối chiếu Bảng I–III](results/PAPER_TABLES_VI.md).
- [168 kết quả CV](results/cv/summary.csv), [fold cố định](results/cv/fold_assignments.csv), [validation](results/validation.json), [so sánh bài báo](results/comparison_paper.csv).
- Notebook `01_data_inspection.ipynb`: cohort và chất lượng dữ liệu.
- Notebook `02_kinematic_features.ipynb`: schema và ma trận động học.
- Notebook `03_pretrained_representations.ipynb`: embedding, GAF/CWT.
- Notebook `04_evaluation_and_fusion.ipynb`: CV, Original Early Fusion và biểu đồ.

Notebook đọc artifact đã có; chọn kernel `.venv`. Kiểm tra bảo toàn mà không huấn luyện:

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/audit_cleanup.py
.venv/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -v
```

Chi tiết refactor và giới hạn kiểm chứng: [bàn giao cleanup](docs/cleanup/REPORT_VI.md).

## Giới hạn tái lập

Tác giả chưa công bố đầy đủ danh sách ID/fold, công thức 74 đặc trưng, định nghĩa nhãn chính xác, chi tiết CWT/encoder bảy kênh và tham số classifier. Cùng 258 người hoặc đạt tỷ lệ tương tự chưa chứng minh triển khai trùng khớp. Nhãn `group` khác cờ dysgraphia ở năm người và cohort có ba người mang cờ omit; các trường hợp này đã ghi trong audit và phân tích nhạy cảm baseline.

F1 của bản dựng là macro; bài báo chưa xác định biến thể F1. Xếp hạng nội bộ CV có thiên lệch chọn mô hình, chưa phải đánh giá trên tập ngoài. Báo cáo baseline ghi rõ khoảng cách với fusion trọng tâm của bài báo; không có kết luận chẩn đoán lâm sàng từ các kết quả này.

## Nghiên cứu Kinematic Task-Level XAI

[Báo cáo tiếng Việt](results/kinematic_xai/REPORT_VI.md) và
[notebook 05](notebooks/05_kinematic_task_xai.ipynb) phân tích riêng Kinematic +
LightGBM trên sáu held-out folds đã lưu. Nghiên cứu gồm TreeSHAP, task/group
importance, grouped permutation, test-time median replacement, Leave-One-Task-Out
retraining và độ ổn định. Baseline được kiểm tra bằng checksum và OOF, không ghi đè.

Output thăm dò ở `results/kinematic_xai/exploratory/`; cross-fitted importance
chỉ dùng Train được lưu riêng ở `results/kinematic_xai/train_only/`. API
`diagramo.kinematic_xai.train_only_task_importance` nhận đúng Train đang xét;
để dùng trong inner CV tương lai phải tính lại trên inner Train.

Môi trường Python 3.10 hiện tại có SHAP 0.49.1. Nếu thiếu các package bổ sung:
` .venv/Scripts/python.exe -m pip install --no-deps -r requirements-kinematic-xai.txt `.
Giữ `requirements-lock.txt` baseline. Lệnh chạy lại, ý nghĩa output space, giới hạn
của task removal và trình tự validation được ghi trong báo cáo. Chỉ xem hoàn tất
khi `results/kinematic_xai/validation.json` có `complete=true`.

## Kinematic Task Transformer

[Báo cáo Task 2](results/task_transformer/REPORT_VI.md) và
[notebook 06](notebooks/06_kinematic_task_transformer.ipynb) so sánh shared Task-MLP,
Transformer nhẹ, motor normalization, Train-only augmentation và XAI-informed task
selection với LightGBM gốc. Dùng lại 258 participants, mapping14×74 và sáu folds cũ;
cấu hình/early stopping chỉ chọn trong inner3-fold. Notebook có bảng số OOF,
fold mean±SD, paired corrected/harmed, robustness, attribution và chi phí.

Code model ở `src/diagramo/task_transformer.py`, raw nuisance ở
`src/diagramo/motor_nuisance.py`, cấu hình ở `configs/task_transformer.json`.
Các script `*_task_transformer*.py` và lệnh tái lập được mô tả trong báo cáo.
Output mới nằm riêng trong `results/task_transformer/`; cache/checkpoints giữ local.
Chỉ coi hoàn thành khi [validation](results/task_transformer/validation.json) có
`complete=true`. “Motor-invariant” là giả thuyết; không có thử nghiệm cross-language.
