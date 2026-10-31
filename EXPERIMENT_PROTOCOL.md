# Tái lập thực nghiệm DiaGraMo

Ngày thiết lập: 2026-10-31. Đây là bản xây dựng lại phương pháp từ mô tả bài báo, không phải mã chính thức của tác giả.

## Nguồn và mục tiêu

- Bài báo đính kèm: *Towards Objective Dysgraphia Detection: A Multi-Branch Deep Learning Approach for Online Handwriting Analysis*, Ouhib và cộng sự, 2026; [bản công khai](https://arxiv.org/abs/2607.09826).
- Dữ liệu cục bộ: `data/clean_data/*_merged.json`; [bản công bố dữ liệu](https://zenodo.org/records/21236910).
- MOMENT dùng API chính thức của [momentfm](https://github.com/moment-timeseries-foundation-model/moment), checkpoint `AutonLab/MOMENT-1-large`.
- Trọng số ảnh dùng torchvision ImageNet; phiên bản và SHA-256 được lưu trong `references/weights_manifest.json`.
- Đã đối chiếu tham khảo [Drotár & Dobeš 2020](https://www.nature.com/articles/s41598-020-78611-9): nghiên cứu đó trích xuất 133 đặc trưng/tác vụ, không cung cấp một danh sách 74 đặc trưng trùng khớp với bài đang tái lập. Vì vậy không coi các công thức dựng lại ở đây là bộ 74 đặc trưng chính thức của tác giả.

## Những gì kiểm tra được trên dữ liệu

294 tệp merged chứa 4.000 bản ghi tác vụ. Có 266 người đủ 14 nhóm tác vụ; 8 người trong số này thiếu nhãn `group`. Giữ 258 người đủ tác vụ và có nhãn nhị phân: 115 nhãn 0, 143 nhãn 1. Vì bài báo không công bố danh sách ID nên việc trùng số lượng 258 chưa chứng minh trùng hoàn toàn cohort của tác giả.

Nhãn chính là `group`, suy luận từ cách chia nhóm có sẵn trong dữ liệu. Cột `special_care_type_dysgraphia` là nhãn khác: có 5 trường hợp không trùng trong 258 người. Không dùng hai nhãn thay thế âm thầm. Có 3 người có `flag_to_omit_total=1` trong cohort; giữ theo quy tắc đủ tác vụ và có nhãn, đồng thời công bố trong bảng audit. Phân tích nhạy cảm theo nhãn/chọn mẫu cần được báo cáo riêng.

Chi tiết 3 cờ loại: BR10306 và BR7311 có `special_care_type_not_relevant_flag_to_omit=1`; BR7406 có `non_czech_language_flag_to_omit=1`. Xem `results/audit/flagged_included.csv`. Đây là lý do không coi việc trùng kích thước 258 là xác nhận cohort hoàn toàn trùng với tác giả.

Có 23.455 bước thời gian không tăng trong toàn bộ dữ liệu và 7 bản ghi có `samples_count` khác độ dài mảng. Kiểm tra mảng thực tế, không cắt mảng theo metadata. Không có mẫu NaN/Inf. Tệp nguồn không bị sửa; SHA-256 từng tệp nằm tại `results/audit/source_hashes.json`.

## Thứ tự dữ liệu

Mỗi hàng là một người, thứ tự ID cố định. 14 nhóm: RCFT recognition, RCFT recall, dictation, big spiral fast, small spiral fast, saw fast, upper loops fast, lower loops fast, rainbow fast, combined loops fast, small spiral precise, lower loops precise, combined loops precise, copy. Gộp dictation 3/4 và copy 3/4 thành cùng nhóm tương ứng; không bỏ hai tác vụ RCFT.

Thứ tự 7 kênh: x, y, pressure, tilt, azimuth, time, pen_status. Chỉ dùng các tín hiệu viết làm đầu vào. Không dùng ID, nhóm, tuổi, giới tính, trường, điểm nhận thức hay cờ chẩn đoán trong đặc trưng.

## Các nhánh

1. **Động học:** 74 đặc trưng/tác vụ, 1.036/người. Đủ số lượng 11 nhóm được bài báo mô tả. Công thức từng đặc trưng không được bài báo công bố; bản dựng này có tên và công thức trong `src/diagramo/features.py`, schema trong `data/derived/schema.json`. Đạo hàm tính theo giây trên mẫu có timestamp tăng nghiêm ngặt; mô tả này là lựa chọn tái lập. Không tính đạo hàm sau khi nội suy còn 512 điểm. Velocity/acceleration tính trên toàn bộ đường đi kể cả in-air; đếm pen lifts từ các đoạn pen_status=0. Không làm trơn hay loại outlier theo nhãn.
2. **MOMENT:** nội suy tuyến tính theo chỉ số mẫu về 512 điểm/kênh; MOMENT-large đóng băng; mean pooling theo kênh và patch mặc định của API chính thức; 1.024/tác vụ, 14.336/người.
3. **GAF:** nội suy tín hiệu về 224 điểm, scale riêng từng kênh/tác vụ về [-1,1], Gramian Angular Summation Field cos(phi_i+phi_j). Kênh hằng được ánh xạ về 0 trước arccos. Đầu vào 7×224×224, đổi khoảng GAF từ [-1,1] sang [0,1].
4. **CWT:** từ tín hiệu 512 điểm, chuẩn hóa từng kênh riêng, wavelet Morlet thực `morl`, 224 scale logspace(1,128), log(1+abs(coefficient)), nội suy trục thời gian về 224, chia cho cực đại từng ảnh. Cache float16; tính biến đổi float32/64. Đây là lựa chọn cụ thể vì bài báo không công bố scale, chuẩn hóa và cách resize.

Ảnh dùng bốn encoder đóng băng: ResNet50 (ImageNet V2, 2.048 chiều), EfficientNet-B4 (V1, 1.792), ViT-B/16 (V1, 768), VGG16 (V1, lớp áp chót 4.096). Chuyển convolution đầu từ 3 sang 7 kênh bằng tổng kernel RGB chia 7 rồi lặp 7 lần; giữ bias. Chuẩn hóa ảnh `(image - 0.449)/0.226` cho tất cả kênh. Các chi tiết này không có trong bài báo, được công bố thay vì giả định là thiết lập chính thức. Không fine-tune encoder trên toàn cohort trước CV. Dùng float32: phép đo trên Quadro T1000 cho thấy float32 nhanh hơn float16 khoảng 5–6 lần; các cache float16 dở dang được tính lại từ đầu. Đầu ra lưu float32.

## Phân loại và fusion

7 bộ phân loại: Logistic Regression, SVM RBF, Random Forest, XGBoost, LightGBM, AdaBoost, MLP. Thêm Dummy theo tần suất lớp để kiểm tra mức nền. Tham số cố định trước khi xem kết quả nhánh tương ứng, lưu tại `results/cv/settings.json`; không tối ưu trên fold kiểm tra.

AdaBoost trong môi trường scikit-learn 1.7.2 dùng SAMME, 100 estimator và learning_rate=0.5. Bài báo không nêu phiên bản thư viện, biến thể AdaBoost hoặc các tham số này; không mặc định kết quả tương đương SAMME.R của các môi trường cũ. Đây là một chi tiết cần có cấu hình tác giả để xác nhận.

| Bộ phân loại | Tham số chính của bản dựng |
| --- | --- |
| Logistic Regression | C=1, max_iter=3000 |
| SVM | RBF, C=1, gamma=scale, probability=True |
| Random Forest | 300 cây, max_features=sqrt, max_depth=None |
| XGBoost | 200 cây, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=4 |
| LightGBM | 200 cây, num_leaves=15, learning_rate=0.05, n_jobs=4 |
| AdaBoost | SAMME, 100 stump, learning_rate=0.5 |
| MLP | 1 tầng ẩn 128, alpha=0.01, max_iter=500, early_stopping=True, validation_fraction=0.15 |
| Dummy | Dự đoán theo prior của train |

Random_state=42 cho các mô hình có tham số này. Các tham số còn lại theo mặc định phiên bản thư viện được khóa trong `requirements-lock.txt`; bản đầy đủ được lưu trong settings JSON.

6 fold StratifiedKFold, shuffle=True, seed=42; mỗi người chỉ nằm ở một fold kiểm tra. Cùng danh sách fold cho mọi cấu hình. Imputer, loại cột hằng và StandardScaler được fit chỉ trên train của mỗi fold. MLP dành 15% train cho early stopping; SVM hiệu chỉnh xác suất bên trong train. Không chọn threshold trên test.

Tối ưu thời gian không đổi mô hình thống kê: tính trước ma trận RBF trong từng fold rồi đưa vào cùng libsvm; đối chiếu trên một fold MOMENT cho nhãn trùng hoàn toàn và sai khác xác suất dưới 1e-14. Chạy các fold độc lập song song; đối chiếu lại Logistic Regression, SVM và LightGBM của nhánh động học cho cùng nhãn dự đoán (sai khác xác suất SVM dưới 2e-14, hai mô hình còn lại bằng 0). Các bước tạo đặc trưng vẫn độc lập với nhãn. XGBoost trong ma trận chính dùng CPU theo thiết lập cố định; phép đo GPU riêng chỉ là benchmark thời gian.

Đánh giá 10 cấu hình đơn nhánh (1 động học, 1 MOMENT, 4 GAF, 4 CWT). Fusion gồm đúng 11 tập con: 6 đôi, 4 bộ ba, 1 bộ bốn của động học/MOMENT/GAF/CWT. Cố định GAF=ViT-B/16 và CWT=EfficientNet-B4 theo nhánh tốt nhất được bài báo công bố; không chọn encoder dựa trên kết quả kiểm tra cục bộ.

Lưu accuracy, F1 macro, F1 weighted, F1 binary, ROC-AUC và balanced accuracy. Báo cáo cả OOF gộp và mean±SD qua 6 fold (SD không phải CI). Bài báo không nêu loại F1 hay cách gộp nên không khẳng định phép so sánh tuyệt đối. Các bảng xếp hạng CV là mô tả nội bộ, có thiên lệch chọn mô hình; chưa phải đánh giá trên tập ngoài độc lập.

## So sánh bài báo

Mốc bài báo: động học/AdaBoost Acc 83,7%, F1 83,6%, AUC 90,1%; GAF/ViT/LightGBM 73,3/73,0/78,5%; CWT/EfficientNet/LightGBM 71,7/71,5/75,3%; MOMENT/SVM 70,2/69,9/74,9%; Kinematic+GAF/AdaBoost 84,9/84,9/91,1%; Kinematic+GAF+MOMENT/AdaBoost 85,3/85,2/90,8%; cả bốn/AdaBoost 83,7/83,7/90,1%.

Các số trên là số công bố, không phải số đã chạy trên máy này. Kết quả thực tế chỉ đọc từ `results/cv/summary.csv` và các tệp dự đoán OOF.

## Giới hạn tái lập

Chưa có danh sách ID/fold, seed, công thức 74 đặc trưng, nhãn đích chính xác, cách xử lý cờ omit, checkpoint/kiến trúc ViT cụ thể, quy tắc 7 kênh, scale CWT, chuẩn hóa ảnh, lớp embedding, tham số classifier và biến thể F1 từ tác giả. Vì vậy đây là tái lập độc lập có ghi nhận giả định; không thể hứa tái hiện chính xác 85,3%.

Riêng CWT của bản dựng được tính từ tín hiệu nội suy 512 điểm. Bài báo chỉ nêu rõ yêu cầu 512 cho MOMENT, chưa mô tả đầy đủ chiều thời gian trước CWT. Lựa chọn này giảm chi phí tính toán nhưng có thể làm mất các biến thiên nhanh trong bản ghi dài; cần mã nguồn tác giả để xác nhận cách xử lý tương ứng. Không xem lựa chọn này là một chi tiết chính thức đã được xác minh của bài báo.
