# Phase 3 recovery — kiểm tra CUDA và ngân sách

**Phase 3 chưa hoàn tất.** Đã xác minh bản sửa ở commit `05b3e28` trên dữ liệu thật, strict deterministic CUDA. Lỗi trước đây nằm ở backward của nội suy tuyến tính; bản sửa dùng toán tử tuyến tính tương đương bằng GEMM, giữ nguyên vị trí các tâm convolution. Không đổi kiến trúc hoặc mục tiêu SSL.

| encoder | windows | finite_gradients | repeated_gradients_bitwise_equal | repeated_forward_bitwise_equal | loss | gradient_l2 |
| --- | --- | --- | --- | --- | --- | --- |
| original | 64 | True | True | True | 0.430542 | 0.417668 |
| multiscale | 64 | True | True | True | 0.617661 | 0.649381 |

Normalizer chỉ fit 143 development-Train participants; dữ liệu smoke lấy từ development Train, không dùng 43 outer Test0. Hai lần forward/backward với cùng weights/input tạo kết quả bit-for-bit bằng nhau. Đây là kiểm tra vận hành, không đo chất lượng embedding hay chẩn đoán.

## Epoch pilot và ước lượng

| quantity | value |
| --- | --- |
| epochs_completed | 1 |
| optimizer_updates | 1 |
| eligible_windows | 11432 |
| training_seconds | 1.219256 |
| forward_only_seconds | 0.219249 |
| windows_per_second | 9376.209754 |
| peak_gpu_allocated_mib | 205.583496 |
| peak_gpu_reserved_mib | 340.000000 |
| estimated_full_train_epoch_seconds | 20.571212 |
| estimated_validation_forward_seconds | 1.863599 |
| estimated_two_seed_pretraining_upper_seconds | 538.435468 |
| estimated_full_phase3_seconds | 3739.487523 |
| configured_full_phase3_budget_seconds | 1800 |
| full_suite_estimate_within_budget | False |

Epoch nhỏ gồm 8 người được lấy theo quantile số window trong development Train; dùng toàn bộ window pen-down hợp lệ của mỗi người, một optimizer update với gradient cân bằng theo participant. Normalizer dùng pool 143 Train. Window pen-up không có reconstruction objective vẫn nằm trong dữ liệu classification; không loại khỏi pipeline classifier.

Đây là một epoch trên tập con, không phải một epoch đầy đủ trên 143 người. Ước lượng theo tỷ lệ window: train epoch khoảng 20.57s; reconstruction validation forward khoảng 1.86s. Hai pretraining seed × tối đa 8 epochs, cộng safety factor 1,5, khoảng 8.97 phút. Validation được benchmark forward-only nên chưa bao gồm toàn bộ baseline/masking/ghi file; không coi đây là thời gian cam kết.

Toàn bộ Phase 3 còn gồm 12 neural downstream fits và 38 CPU readout fits. Dùng thời gian downstream duration-aware đã đo để proxy, tổng khoảng 62.32 phút; chưa cộng đủ CPU/extraction và uncertainty. Ngân sách riêng đặt cho full Phase 3 là 30 phút để dành phần còn lại của phiên cho validation nhẹ. Không khởi chạy full SSL. Trạng thái giữ **pilot-only/incomplete**, dù pretraining riêng lẻ có thể khả thi.

Checkpoint pilot: `data/derived/raw_temporal_v2/phase3_recovery/pilot_epoch1.pt` (cache cục bộ), SHA `4f519305e644ca512b843e37cc66c84099737f5c201aabc42f913b0fcc70fb7a`. Không dùng checkpoint này để tạo embedding toàn cohort hoặc đưa vào CV. Chuẩn hóa/encoder đã thấy pool 143 người nên không được tái sử dụng tùy tiện cho các fold khác.


Khuyến nghị: ưu tiên OOF của mô hình scratch đã có tín hiệu development trước. Chỉ lên lịch full SSL sau khi thấy nhu cầu bổ sung thông tin và khóa ngân sách cho cả matched scratch/random/frozen/fine-tune controls; reconstruction loss đơn lẻ không đủ để khẳng định embedding hữu ích.
