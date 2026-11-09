# Diagnostic attempt — not final XAI results

Lượt kiểm tra ban đầu dừng khi linear logit attribution CWT sai số 8.498e-8 do
phép trừ float32 trước khi nhân hệ số float64. Không tăng tolerance; chuyển các
phép tính decomposition sang float64 và thêm regression test. Không dùng các
artifact ở đây cho bảng/hình/kết luận cuối. Không sửa checkpoint hoặc dự đoán gốc.
