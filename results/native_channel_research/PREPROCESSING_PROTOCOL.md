# Tiền xử lý native numeric

Read-only Phase6 reader, giữ row order và mọi mẫu hợp lệ. Fit first/second moments chỉ từ Train: equal participants, equal recordings trong participant, equal samples trong recording. Không dùng label trong API preprocessing. StandardScaler B1/B2 fit Train.

Normalize trước fixed permutation. Non-overlap windows256; zero padding sau scaler và explicit valid mask. Không cross-recording; pooling window theo valid sample count, trung bình recordings, loss participant. Giữ terminal samples; không sort c2 hoặc tạo task identities.

B0 tái sử dụng năm checkpoint prior Phase 6, không refit. B1 có sáu thống kê số recording hợp lệ và số mẫu. B2 có 105 features: 77 thống kê marginal và 28 thống kê absolute successive differences. Vì vậy **B2 không hoàn toàn bất biến với thứ tự**. LightGBM cố định 100 trees, 7 leaves, max_depth=3, learning_rate=0,03, min_child_samples=12, reg_lambda=1. StandardScaler chỉ fit Train. Không search hoặc chọn feature bằng Test.
