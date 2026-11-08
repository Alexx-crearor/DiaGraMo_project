# Thiết kế Fusion và bản đồ code

Thiết kế được cố định trong `settings.json` trước lượt chạy, gồm SHA-256 mã và cache.
Tệp này giải thích các lựa chọn; trạng thái thực nghiệm chỉ lấy từ `validation.json`.

## Tận dụng project hiện có

- `prepare.py` và `features.py`: cohort 258 người đủ 14 tác vụ, nhãn `group`, Kinematic 1.036 chiều.
  Đây là bộ đặc trưng dựng lại có ghi nhận giả định, không xác nhận công thức chính thức của bài báo.
- `extract.py`: MOMENT, GAF/ViT-B16 và CWT/EfficientNet-B4 gốc; encoder đã đóng băng.
- `extract_embedding_ablation.py`: GAF channel_concat và adapter cố định, không đọc nhãn.
  Chỉ dùng lại ma trận đã hoàn tất và được kiểm tra hash; không chạy extraction trong nhiệm vụ này.
- `evaluate.py`: classifier và metrics gốc; `build_pipeline` giữ nguyên hành vi SVM/RBF precomputed.
- `reduction.py`/Feature Optimization: chỉ dẫn tập reducer nhỏ; không tái dùng reducer đã fit.
- Embedding Ablation: chỉ dẫn chọn GAF channel_concat/adapter và CWT gốc làm đối chứng;
  không suy rằng GAF đơn nhánh tốt hơn tự động làm Fusion tốt hơn.

## Bốn phương pháp và đối chứng

Giữ nguyên Kinematic + LightGBM làm baseline cố định. Các tổ hợp chính chỉ gồm KM, KG, KC, KMG.
Thêm KG_adapter/KMG_adapter để đối chiếu chi phí GAF; không mở rộng thêm tổ hợp.

1. Original Early nối embedding cũ, chọn classifier trong inner Train.
2. Compact Early chọn reducer riêng từng nhánh trước khi nối; Kinematic không giảm chiều.
3. Late trộn xác suất với trọng số simplex; đối chiếu uniform pooling.
4. Adaptive giới hạn tổng đóng góp nhánh phụ 25%/50%, gate theo độ bất định Kinematic;
   cho phép trả về đúng Kinematic nếu inner mean F1 không tăng hoặc inner Accuracy giảm.

Lưới nhỏ gồm MOMENT ANOVA-64/PCA-99%, GAF ANOVA-64/256/PCA-95%, CWT none/PCA-99%.
Ba classifier đại diện giữ tham số cũ: LR, SVM, LightGBM. Kinematic luôn dùng LightGBM khi là
base model trong Late/Adaptive. Không dùng kết quả outer Test để chọn tham số hay threshold.
`historical_early_summary.csv` giữ riêng kết quả cả bảy classifier lịch sử, không đánh đồng
quy trình nested chọn trong ba classifier với hàng thắng bảng cũ.

## Ba mức chia dữ liệu

Outer Train 215 người / Test 43 người, đúng assignments cũ, seed 42. Trong outer Train:

1. Inner 3-Fold chia Train/Validation để chấm từng ứng viên.
2. Để huấn luyện combiner của một inner split, cross-fit 3-fold **bên trong inner Train**.
   Toàn bộ imputer/scaler/PCA/ANOVA/base classifier được refit tại mức này.
3. Combiner fit trên xác suất cross-fitted và nhãn inner Train, chấm trên inner Validation.
4. Chọn theo ba inner Validation, refit base models trên outer Train. Outer combiner fit từ
   cross-fitted probabilities của ba inner split chính, sau đó nhận xác suất outer Test.

Base options được cố định theo ứng viên trước khi tạo xác suất cross-fitted. Không chọn một
base model bằng tất cả nhãn outer Train rồi giả vờ xác suất cùng cohort là độc lập với lựa chọn đó.
Inner Validation dùng để chọn toàn bộ cấu hình; bước đánh giá combiner trong mỗi inner split
vẫn chỉ học từ inner Train. Outer Test labels chỉ xuất hiện trong bước metrics.

## Chi phí và quản lý kết quả

`BranchBank` chỉ fit các reducer cần thiết, chia sẻ tính toán trong cùng split, không fit
ExtraTrees hay PCA cho nhánh không yêu cầu. Pipeline dùng CPU và tối đa hai outer jobs đồng thời.
Không có fine-tuning encoder hoặc yêu cầu VRAM mới. Checkpoint outer được lưu ở máy trong thư mục
này và gitignored vì kích thước lớn. Bản sao code, bảng số, OOF và audit có thể commit riêng.

Mỗi stage có train/test IDs, reducer metadata/selected feature indices, hash và chi phí.
`report_fusion_ablation.py` tái dựng split, replay dự đoán inner và checkpoint outer, tính lại
metrics, xác minh baseline và artifact cũ. `validate_fusion_ablation.py` chỉ đánh dấu hoàn tất
sau khi số liệu, notebook đã execute, PNG/SVG và unit tests đều hợp lệ.

Không có đảm bảo Fusion tăng điểm Test, kể cả khi safety rule trên Train được thỏa mãn.
Thiết kế đã tham khảo cùng cohort ở các vòng trước; cần cohort ngoài để xác nhận mức tăng nhỏ.
Không thực hiện XAI hoặc tự động thay mô hình baseline đang sử dụng.
