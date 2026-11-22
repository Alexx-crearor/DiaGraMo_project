# Tóm tắt Phase 7

| Model | Macro-F1 | AUC | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|
| B0 | 0.344262 | 0.479950 | 0.500000 | 0.525000 |
| B1 | 0.598214 | 0.592175 | 0.604845 | 0.600000 |
| B2 | 0.932413 | 0.960457 | 0.929825 | 0.933333 |
| B3 | 0.941337 | 0.969368 | 0.940267 | 0.941667 |
| B4 | 0.941337 | 0.971039 | 0.940267 | 0.941667 |

B3−B4: ΔMacro-F1 = 0.000000; paired CI95% [0.000000, 0.000000]. Corrected=0, harmed=0; số dự đoán nhị phân khác nhau=0. ΔAUC=-0.001671.

CNN có F1 nhỉnh hơn thống kê trong lần chạy này, nhưng không hơn đối chứng hoán vị. Không có bằng chứng lợi ích của thứ tự hàng theo phép kiểm tra đã khóa. Điểm cao có thể liên quan đến phân bố giá trị, quan hệ giữa các cột hoặc nguồn thu thập; thí nghiệm này không xác định đóng góp nhân quả của từng yếu tố.

Decision B; opaque suite complete=True.
