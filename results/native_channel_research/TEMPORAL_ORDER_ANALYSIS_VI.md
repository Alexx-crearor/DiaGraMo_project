# Phân tích thứ tự quan sát

B3−B4: ΔMacro-F1 = 0.000000; paired CI95% [0.000000, 0.000000]. Corrected=0, harmed=0; số dự đoán nhị phân khác nhau=0. ΔAUC=-0.001671.

B3−B4: ΔMacro-F1 = 0.000000; paired CI95% [0.000000, 0.000000]. Corrected=0, harmed=0; số dự đoán nhị phân khác nhau=0. ΔAUC=-0.001671.

CNN có F1 nhỉnh hơn thống kê trong lần chạy này, nhưng không hơn đối chứng hoán vị. Không có bằng chứng lợi ích của thứ tự hàng theo phép kiểm tra đã khóa. Điểm cao có thể liên quan đến phân bố giá trị, quan hệ giữa các cột hoặc nguồn thu thập; thí nghiệm này không xác định đóng góp nhân quả của từng yếu tố. Nếu hard predictions giống nhau, CI bootstrap F1 bằng [0,0] là hệ quả tất định của các dự đoán cố định này; không chứng minh tương đương trên quần thể, seed khác hoặc lần train khác.

2.000 paired IID participant bootstrap, seed42, cùng resampled indices và fixed OOF cho mọi model. CI không bao gồm refitting, seed variance, source clustering, cohort selection hoặc researcher-selection uncertainty. Exploratory multiple comparisons không được hiệu chỉnh. ERROR_OVERLAP.csv có corrected/harmed/both-correct/both-wrong. ERROR_BY_LENGTH.csv dùng quartiles độ dài cố định và Spearman mô tả, không phải causal shortcut test. Không coi 2,17 triệu mẫu là independent participants.
