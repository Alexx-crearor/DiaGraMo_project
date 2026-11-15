# Bàn giao: khôi phục bản tái lập bốn nhánh DiaGraMo

Ngày: 2026-11-16 (Asia/Bangkok).

Branch: `cleanup/original-four-branch-reproduction`.
Commit trước cleanup: `273fc9b742d369816023a33c24c75709a56d0bfc`.
Mốc baseline dùng để đối chiếu nguồn: `ac4f927`.

Repository hiện tập trung vào bản tái lập độc lập ban đầu: Kinematic, MOMENT,
GAF, CWT; chuẩn bị dữ liệu, trích xuất, phân loại, Original Early Fusion,
đánh giá và báo cáo. Các giả định và giới hạn tái lập trong protocol được giữ nguyên.

## Kiểm toán và quyết định

[classification.json](classification.json) phân loại từng đường dẫn tracked và
tệp ghi chú người dùng thành BASELINE, IMPROVEMENT, SHARED hoặc UNKNOWN.
[dependencies_before.json](dependencies_before.json) ghi import graph trước cleanup;
[structure_validation.json](structure_validation.json) ghi graph và kiểm tra sau cleanup.

| Thành phần | Phân loại/quyết định | Cơ sở |
|---|---|---|
| prepare, download_weights, run_all, report, report_paper_tables, sensitivity, train_selected | BASELINE, giữ | Có trong commit tái lập đầu tiên; vẫn nhập và đọc artifact baseline |
| features.py | SHARED, giữ nguyên công thức và API | Chuẩn bị dữ liệu và biến đổi ảnh baseline dùng trực tiếp |
| kernels.py | SHARED, giữ nguyên | SVM baseline dùng RBF precomputation |
| evaluate.py | SHARED, giữ classifier, CV, Early Fusion, build_pipeline | Helper được tách trong giai đoạn sau nhưng baseline hiện dùng; test khớp tham số đã lưu |
| extract.py | SHARED, giữ baseline bảy kênh | Bỏ tùy chọn RGB chỉ dành cho ablation; AST hàm load_encoder khớp baseline đầu tiên sau khi bỏ docstring |
| execute_notebooks.py | SHARED, giữ | Glob chỉ còn bốn notebook; --only và --validation-output vẫn dùng được; không có danh sách notebook cũ cần sửa |
| reduction, channel_embedding, fusion, explain và tests đi kèm | IMPROVEMENT, xóa | Không có import từ pipeline baseline; original concatenation nằm trong evaluate.py |
| audit_embedding_baseline.py và ba common helper | IMPROVEMENT, xóa | Ghi vào output ablation và phụ thuộc script ablation; không cần cho baseline |
| Ghi chú review cục bộ | UNKNOWN, giữ ngoài commit | Tệp người dùng có từ trước, không tự ý sửa/xóa |

Đã xóa **1.055 tệp tracked**: bốn thư mục kết quả cải tiến, năm notebook 05–09,
các script/module/test phục vụ cải tiến và requirements riêng. Danh sách chính xác:
[removal_plan.json](removal_plan.json). Có **24 artifact cục bộ ngoài Git**, tổng
khoảng **1,245 GiB**, trong các thư mục embedding/fusion ablation đã kiểm kê và xóa.
Chúng gồm embedding/cache tác vụ và checkpoint của thực nghiệm cải tiến.
Không xóa tệp nguồn trong data hoặc pretrained weights.

[changes.json](changes.json) liệt kê từng tệp xóa, thêm và sửa. **Không di chuyển
tệp baseline nào.** Các thực nghiệm tracked trước cleanup vẫn có trong lịch sử
commit đã ghi ở trên; artifact cải tiến ngoài Git không nằm trong lịch sử đó.

## Tổ chức bốn nhánh và hành vi

```text
prepare.py -> branches.kinematic.extract -> features.kinematic
extract.py -> branches.moment.extract -> MOMENTPipeline(...).embeddings
extract.py -> branches.gaf.transform -> features.gaf -> frozen image encoder
extract.py -> branches.cwt.transform -> features.cwt -> frozen image encoder
evaluate.py -> concatenation -> imputer / variance filter / scaler -> classifier
report.py / report_paper_tables.py -> baseline CSV / JSON / OOF
```

Package `src/diagramo/branches/` cung cấp bốn điểm truy cập rõ ràng. Kinematic,
GAF và CWT gọi đúng function object cũ; MOMENT giữ nguyên lời gọi `model(x_enc=x).embeddings`.
`prepare.py` và `extract.py` dùng các điểm truy cập này. Công thức, nội suy,
normalization, dtype, inference context, pooling và thứ tự vector không được viết lại.

Source thay đổi theo hướng tổ chức và bỏ API RGB của ablation. CLI baseline giữ
nguyên. `build_pipeline` giữ nguyên logic; chỉ sửa docstring cho đúng phạm vi.
API trong `diagramo.features` vẫn tồn tại cho notebook và script cũ.
Toàn bộ thân hàm `load_encoder` được kiểm tra AST với phiên bản baseline đầu tiên.
[source_validation.json](source_validation.json) ghi SHA nguồn và kết quả đối chiếu.

[repository_tree.txt](repository_tree.txt) cung cấp cây thư mục sau refactor.
README mới mô tả dữ liệu, bốn nhánh, classifiers, Original Early Fusion, môi
trường, chạy baseline và đọc kết quả.

## Bảo toàn artifact baseline

Manifest được tạo trước khi dọn artifact:
[baseline_manifest.json](baseline_manifest.json), gồm path, bytes và SHA-256.
Sau cleanup, kiểm tra lại **12.387 tệp**, tổng **19.750.156.592 byte**.
[preservation_summary.csv](preservation_summary.csv) ghi đầy đủ từng nhóm.

| Nhóm | Số tệp | Xác nhận |
|---|---:|---|
| results/cv | 339 | 168 JSON + 168 OOF CSV + summary/settings/fold assignments; SHA-256 nguyên vẹn |
| results/audit | 5 | Nguyên vẹn |
| results/sensitivity | 7 | Nguyên vẹn |
| results/figures | 7 | Nguyên vẹn |
| results/models cục bộ | 51 | Nguyên vẹn |
| Metadata/báo cáo gốc tại results | 13 | Nguyên vẹn |
| Notebook gốc 01–04 | 4 | Nội dung và output gốc nguyên vẹn |
| data/clean_data | 294 | Nguyên vẹn |
| data/derived | 3.652 | Nguyên vẹn, gồm embedding và cache baseline |
| data/processed_data và data/raw_data | 8.004 | Nguyên vẹn |
| .cache/huggingface và .cache/torch | 8 | Nguyên vẹn |
| references, protocol, requirements-lock | 3 | Nguyên vẹn |

Đã đối chiếu 258 ID duy nhất, nhãn và fold của từng OOF với assignments; tái tạo
StratifiedKFold(6, shuffle=True, random_state=42) chỉ để so sánh assignments.
Đã tính lại cả sáu metric từ **dự đoán đã lưu**, đối chiếu JSON và summary với
sai số tuyệt đối tối đa cho phép 1e-12; kiểm tra cả metric từng fold. Không fit
classifier trong bước này. [artifact_validation.json](artifact_validation.json)
xác nhận 168/168 kết quả, OOF và tất cả metric. `results/validation.json` gốc
vẫn `complete=true`, 168/168 và giữ nguyên checksum.

## Kết quả kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| Python compile/AST và import graph | Đạt; không import module đã xóa |
| Import 10 script baseline và bốn module nhánh | Đạt |
| CLI --help của extract/evaluate/run_all/execute_notebooks | Đạt |
| Dependency pip check | No broken requirements found |
| Unit tests | 6/6 đạt; log tại tests.log |
| Ma trận cache | Đủ 10 biểu diễn, 258 hàng, đúng số chiều, ID/nhãn và hữu hạn |
| Kinematic từ tín hiệu nguồn | 42 tác vụ thuộc ba participant đầu/giữa/cuối khớp từng phần tử với cache |
| GAF/CWT | Biến đổi CPU trả 7×224×224 hữu hạn, gọi đúng hàm cũ |
| MOMENT | API trả đúng embeddings; khởi tạo offline từ revision đã pin |
| Bốn image encoders | Khởi tạo CPU từ cache; đúng số chiều, đóng băng; kernel/bias khớp chính xác baseline |
| Original Early Fusion | Đủ 11 tổ hợp × 8 classifiers, đúng feature keys và số chiều |
| Báo cáo baseline | Hai script chạy trên bản sao; tám CSV sinh lại khớp nội dung số; validation JSON khớp |
| Bốn notebook | Thực thi bản sao thành công: 5 + 4 + 4 + 7 = 20 code cells; không có cell lỗi |

Chi tiết: [encoder_validation.json](encoder_validation.json),
[report_validation.json](report_validation.json),
[notebook_validation.json](notebook_validation.json),
[validation.json](validation.json). Bản sao chạy thử ở `tmp/`, được Git ignore;
không ghi đè báo cáo, notebook hoặc requirements-lock gốc.

## Chạy lại và kiểm tra

Các lệnh tái lập baseline từ đầu được ghi trong README. Các kiểm tra riêng của cleanup:

```powershell
.venv/Scripts/python.exe -X utf8 -B scripts/audit_cleanup.py
.venv/Scripts/python.exe -X utf8 -B scripts/validate_baseline.py structure
.venv/Scripts/python.exe -X utf8 -B scripts/validate_baseline.py encoders
.venv/Scripts/python.exe -X utf8 -B scripts/validate_baseline.py reports
.venv/Scripts/python.exe -X utf8 -B scripts/validate_baseline.py notebooks
```

Lệnh encoders chỉ khởi tạo trên CPU; lệnh reports và notebooks chỉ ghi bản sao
trong tmp. Notebook kernel cần runtime Jupyter ghi được trên Windows.

## Giới hạn kiểm chứng

Không huấn luyện lại classifier hoặc chạy forward pretrained encoder để tạo
embedding mới. Tính nguyên vẹn artifact cũ, Kinematic mẫu, công thức dùng chung,
khởi tạo encoder, tham số và đường đi dữ liệu đã được kiểm chứng; chưa kiểm tra
lại số học end-to-end của toàn bộ encoder/CV sau refactor. Không tuyên bố một
lượt thực nghiệm mới tương đương số học đã hoàn tất.

Protocol và giới hạn của bản tái lập độc lập tiếp tục áp dụng: chưa có đầy đủ ID,
fold, công thức, checkpoint và cấu hình gốc của tác giả. Công việc này không
cung cấp bằng chứng cải thiện Accuracy hay tái hiện chính xác triển khai tác giả.

Môi trường cục bộ vẫn giữ các package đã cài; `requirements-lock.txt` baseline
không thay đổi. Transformers/MOMENT phát cảnh báo deprecation/head khi khởi tạo;
Jupyter có cảnh báo TCP transport. Các bước kiểm tra hoàn thành và không có
notebook cell lỗi. Tệp ghi chú review có sẵn của người dùng được giữ untracked.

Commit chỉ được tạo trên branch cleanup. Không push hoặc merge vào main trong
nhiệm vụ này.
