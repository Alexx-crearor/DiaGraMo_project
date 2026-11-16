# Kiểm toán dữ liệu Raw Temporal Graphomotor

Đã kiểm tra toàn bộ 258 participants, 3,612 recordings và 54,808,740 samples. Giữ đúng nhãn `group`, 115 người group0 và143 người group1; không đổi sang diagnosis flag. Không loại thêm participant hoặc task. Source keys, raw sample counts, hash JSON gốc và run inventories đều được lưu.

## Schema, đơn vị và thứ tự

Nguồn có đúng x, y, time, pen_status, pressure, azimuth, tilt. Thứ tự channels của source schema khác thứ tự lưu trong dictionary: pipeline lấy theo tên, không đoán vị trí. Original time được xử lý theo đơn vị số gốc, quy ước giây như baseline; không có chứng chỉ calibration thiết bị. Tọa độ và pressure giữ đơn vị gốc của thiết bị; không tự đổi thành mm hoặc Newton. Azimuth/tilt được giữ giá trị số gốc, không giả định calibration góc ngoài dữ liệu.

Không sort theo time, không drop duplicate, không resample sang512 điểm, không dùng engineered74 features. Mỗi recording là chuỗi acquisition-order có mapping raw index→timestamp→pen-state run→anomaly-split segment→window. Các task index là identity, không phải chronological ordering của14 assessments.

## Kết quả audit

| task | recordings | samples_median | samples_max | duration_median | duration_max | median_effective_hz | dt_cv_median | duplicates | large_gaps |
|---|---|---|---|---|---|---|---|---|---|
| RCFT-recall | 258 | 15574 | 94835 | 126.5 | 579.72 | 127 | 12.981 | 1942 | 2645 |
| RCFT-recognition | 258 | 25627 | 105525 | 221.35 | 2510.7 | 117.64 | 14.019 | 2029 | 5714 |
| big-spiral-fast | 258 | 1277.5 | 9608 | 6.8515 | 51.6 | 187.33 | 0.088624 | 6 | 38 |
| combined-loops-fast | 258 | 2868 | 15641 | 16.29 | 85.763 | 187.33 | 0.088998 | 60 | 88 |
| combined-loops-precise | 258 | 3516 | 8643 | 20.888 | 71.416 | 187.33 | 0.089675 | 29 | 85 |
| copy | 258 | 55697 | 155103 | 392.87 | 1231.7 | 146.88 | 13.065 | 6924 | 5551 |
| dictation | 258 | 82326 | 199739 | 565.49 | 1216.7 | 149.93 | 13.777 | 10160 | 6322 |
| lower-loops-fast | 258 | 1929.5 | 6144 | 10.383 | 39.593 | 187.33 | 0.088656 | 53 | 67 |
| lower-loops-precise | 258 | 2723.5 | 6240 | 14.917 | 35.732 | 187.33 | 0.088625 | 27 | 60 |
| rainbow-fast | 258 | 1223.5 | 4190 | 6.8755 | 26.661 | 187.33 | 0.088621 | 8 | 33 |
| saw-fast | 258 | 909.5 | 2558 | 4.909 | 16.442 | 187.33 | 0.088624 | 6 | 12 |
| small-spiral-fast | 258 | 1168.5 | 4689 | 6.2505 | 33.91 | 187.33 | 0.088621 | 10 | 24 |
| small-spiral-precise | 258 | 2685 | 22994 | 15.242 | 140.96 | 187.33 | 0.088621 | 23 | 58 |
| upper-loops-fast | 258 | 1100.5 | 3431 | 6.021 | 21.351 | 187.34 | 0.088619 | 6 | 28 |

Toàn cohort có 21,283 Δt=0; 0 Δt<0; 0 khoảng0<Δt<0,0001; 20,725 khoảng>1 giây. Nonfinite samples=0. Missing channels=0 và độ dài bảy channels khớp trong mỗi recording. Time không đều: effective Hz từ toàn duration khác1/median positive Δt, vì vậy lưu cả hai và coefficient of variation; không dùng index difference làm physical elapsed time.

Pen-state runs (state1 là pen-down, state0 là pen-up theo source/baseline convention):

| pen_state | runs | samples_median | duration_median | duration_p95 | duration_max |
|---|---|---|---|---|---|
| 0 | 112218 | 104 | 0.662 | 6.7971 | 2186.8 |
| 1 | 115830 | 167 | 0.883 | 5.845 | 48.501 |

Một pen-down run được xem là stroke theo định nghĩa vận hành, không phải nét chữ có ground-truth ký tự. Timestamp anomalies có thể chia một run thành nhiều segment an toàn; lưu cả original run index và segment index. Pen-up pauses vẫn được giữ, không ghép qua chúng để tính vận tốc.

## Chính sách timestamp và đạo hàm

| Trường hợp | Xử lý và thông tin giữ lại |
|---|---|
| Sample đầu | Không có previous interval; derivative invalid, không đánh dấu duplicate giả |
| 0,0001 ≤ Δt ≤1 và cùng pen-down run | Backward velocity dx/Δt,dy/Δt; acceleration chỉ khi hai velocity intervals liên tiếp đều valid |
| Δt=0 | Giữ sample và time gốc; boundary mới; dt_valid0, duplicate flag1; velocity/acceleration0 cùng validity mask0 |
| Δt<0 | Giữ acquisition order, không sort/đoán clock reset; boundary mới, negative flag1, derivative invalid; thực tế cohort không có |
| 0<Δt<0,0001 | Boundary/invalid derivative để tránh nổ số; giữ tiny flag; cohort hiện không có |
| Δt>1 giây | Giữ gap gốc trong timestamp/elapsed metadata, boundary mới; không suy vận tốc liên tục qua gap |
| Pen state thay đổi hoặc pen-up | Không tính motion derivative qua transition hoặc trên pen-up; giữ tọa độ, pressure, thời gian, state và masks |

Các ngưỡng cố định là chính sách sensor-safety, không chọn từ Test labels. Chúng có thể loại đạo hàm ở chuyển động rất chậm hoặc pauses hữu ích; model vẫn thấy timing và anomaly flags. Chưa có ground truth clock/sensor để khẳng định mọi gap là lỗi. Velocity/acceleration dùng actual Δt, không giả định sampling uniform. Không đưa jerk vào phiên bản đầu do nhạy với đạo hàm bậc cao và timestamp anomalies.

## Biểu diễn neural

21 sample channels: relative x/y (trừ điểm đầu, giữ kích thước), pressure, azimuth, tilt, elapsed source time, valid Δt, pen state, vx/vy/speed, ax/ay, velocity/acceleration/dt validity, duplicate/negative/tiny/large-gap indicators, boundary. Raw values và index/time mapping vẫn truy xuất từ original JSON. Relative position làm mất absolute page placement; không xóa isotropic scale hoặc duration. Signed log1p cho continuous channels giảm dynamic range, có thể làm nhỏ khác biệt biên độ; flags giữ nguyên.

Train-only normalization kết hợp per-participant moments với trọng số mỗi người bằng nhau, rồi áp dụng channel mean/SD; fixed SD floor0,1 và z saturation±10. Không fit normalization từ toàn cohort. Statistics per-person lưu trước nhưng chỉ gộp fit IDs. Raw64-bit timestamps không qua saturation. Index-time control dùng dx/d(sample index), không gọi đó là vận tốc vật lý; neutralize clock-valued channels/flags và dùng sample-order encoding, nhưng vẫn giữ boundary inventory để không phá integrity.

## Segmentation và tài nguyên

Stroke/anomaly-aware non-overlapping windows tối đa128 raw samples. Long runs chia nhỏ, run ngắn padding có mask; không random crop, không truncate, không concatenate qua pen-state boundary. Tổng 54,808,740/54,808,740 samples được phủ đúng một lần. Có 567,704 windows, padding fraction 24.57%. Physical duration mỗi window khác nhau, không giả định128 points là cùng số giây.

So sánh chi phí window sizes, chỉ từ inventory không dùng nhãn:

| patch_size | patches | raw_samples | padded_cells | all_channels_float32_GiB |
|---|---|---|---|---|
| 64 | 987787 | 54808740 | 8409628 | 4.9456 |
| 128 | 567704 | 54808740 | 17857372 | 5.6847 |
| 256 | 373333 | 54808740 | 40764508 | 7.4768 |

Chọn128 cân bằng chi phí, không chọn bằng Test performance. Window64 tăng số tokens và overhead;256 gộp dài hơn có thể mất local detail. Đây là trade-off tính toán, chưa là kết quả predictive ablation về window size.

CNN dùng kernel9 stride8 có vùng phủ chồng lấn, không bỏ positions giữa receptive fields; sau đó convolution3 và masked pooling. Bounded attention tối đa16 windows/stroke chunk và32 segments/task chunk; các chunk đều được tổng hợp, không bỏ phần cuối. Chunk boundaries hạn chế long-range interactions, cần công khai trong báo cáo. Original stroke và chunk boundaries được phân biệt.

Raw float32 bảy channels cần khoảng1.429GiB; hai banks padded21channels float32 cần khoảng11,37GiB. Đã giữ cachefloat32 gốc và tạo packed float16 storage khoảng5,68GiB để chia sẻ qua memory map; model/gradient vẫn float32, không dùng AMP. Max quantization error đo trên mọi input log-channel là0,00390625; timestamps/inventories vẫn float64. Pilot được thực hiện chỉ trên outer Train, không dùng Test; peak tensor memory và latency ghi riêng, không đồng nhất với toàn bộ GPU driver allocation.

Không có kết luận learning/accuracy từ data audit. Attribution spike, pause dài hoặc đạo hàm cao không chứng minh tremor, dysgraphia hoặc cơ chế sinh học.
