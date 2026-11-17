# Quy ước resource budget

Config gốc `configs/raw_temporal_v2/resources.json` giữ nguyên để bảo toàn chữ ký thực nghiệm.

- `maximum_training_runs=360`: budget các fits neural/SSL do GPU scheduler quản lý. Ledger dùng `running/completed/failed`; có tối đa một GPU worker. Resume khớp chữ ký giữ cùng experiment ID.
- `maximum_cpu_probe_fits=600`: budget estimator `.fit()` trên CPU, bao gồm inner fits của physical probes. Phase1 gồm380 fits cho38 cấu hình probe, mỗi cấu hình9 inner fits và1 final fit. Các readout tiếp theo được ghi trong `cpu_fit_budget.json`.
- CPU probe/readout trong ledger dùng `cpu_running/cpu_completed/cpu_failed`, cùng `completion_status`. Tách trạng thái này để scheduler GPU đã khóa không tính CPU probe thành neural fit. Không có nghĩa CPU fits chưa huấn luyện; số fit thực tế được ghi rõ trong `training_cost` và validation từng phase.
- Hai engineering/audit jobs Phase0 có trạng thái `completed` cũng được scheduler trừ bảo thủ vào budget360. Chúng được phân biệt với30 neural fits thực tế khi thống kê; không tăng budget để bù các dòng này.
- Các LightGBM/meta-classifier CPU trong Phase4 là classification/fusion fits, không phải embedding probes; chặn riêng bằng `other_cpu_maximum_fits=126` trong config Phase4, ghi ở `cpu_classification_fit_budget.json`. Checkpoint LightGBM cũ tái sử dụng không tính là fit mới. Tổng thực tế của mọi loại fit phải được báo cáo, không gọi tổng này là số neural fits.
- Dòng kế hoạch tổng hợp có trạng thái `expanded_to_child_runs`, không tính là fit thứ hai; completion của dòng tổng hợp chỉ được xác nhận sau validation các child runs.
- Classification inference giữ toàn bộ window/source coverage. Masked SSL chỉ đặt objective ở target pen-down hợp lệ; pen-up-only window không có target tái dựng, được đếm riêng, không loại khỏi classification.
- Báo cáo peak allocated GPU tensor memory, không đồng nhất với tổng process memory của `nvidia-smi`. Không chạy nhiều GPU training worker trên T1000 4GB.
- Không dùng outer Test để giảm budget, chọn epoch, chọn kiến trúc hoặc trọng số. Nếu budget hết, kết quả chưa chạy phải để skipped/not_executed, không gán completed.

Thời gian chưa được giới hạn tổng (`wall_clock_hours=null`), free storage tối thiểu20GiB và giới hạn GPU tensor3500MiB.
