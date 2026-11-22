# Giao thức khóa trước điểm số

LOCKED_CONFIG.json và PREEXPERIMENT_DECISIONS.json được ghi trước điểm số mới. Five-fold participant CV giữ nguyên Phase 6, base seed42; code cố định initialization/data-order seed=42+fold cho cả B3/B4, permutation seed42. Đây là một suite với cùng quy tắc seed, không phải chọn seed tốt nhất. Threshold p>0,5, final epoch20, AdamW lr=1e-3/wd=1e-3, clip1, batch8 participant bằng accumulation. Không outer-Test early stopping, threshold tuning, architecture search hoặc seed selection. Quyết định không chạy sensitivity suite base seed43 được khóa trước score. SHA config: `d895cff2f4613c989c660431cde2f85555b133f634ea053c452d7da55ce3e24b`.

B0 tái sử dụng năm checkpoint prior Phase 6, không refit. B1 có sáu thống kê số recording hợp lệ và số mẫu. B2 có 105 features: 77 thống kê marginal và 28 thống kê absolute successive differences. Vì vậy **B2 không hoàn toàn bất biến với thứ tự**. LightGBM cố định 100 trees, 7 leaves, max_depth=3, learning_rate=0,03, min_child_samples=12, reg_lambda=1. StandardScaler chỉ fit Train. Không search hoặc chọn feature bằng Test.

13.761 tham số; ba Conv1D song song kernels 5/13/29, mỗi nhánh 32 channels; ReLU, masked pooling, Linear96→32, ReLU, dropout0,1 và Linear32→1. Không attention, SSL, pretrained model hoặc original 14-task head. Window256 không chồng lấp; pooling theo số mẫu hợp lệ rồi trung bình recording. Loss tính theo participant.

B3/B4 cùng kiến trúc, tham số khởi tạo, folds, optimizer, effective batch và 20 epochs. B4 hoán vị đồng bộ bảy cột trong từng window sau normalization; SHA-256 key cố định từ seed42/participant/recording/window-start. Không hoán vị padding, không mix người, không mất mẫu, không chọn permutation theo score. Dù thứ tự hàng bị phá, c2 có thể còn mang dấu vết vị trí; không tuyên bố xóa sạch mọi thông tin vị trí tiềm ẩn.

Resource admission: training-only measured throughput; dự toán đủ cả matched B3+B4. OOM fallback chunk64→32→16→8→CPU, giữ architecture và effective batch8. Checkpoint epoch model/optimizer/RNG; resume không refit completed folds. Văn bản được sửa encoding UTF-8 sau chạy; LOCKED_CONFIG và recipe không thay đổi.
