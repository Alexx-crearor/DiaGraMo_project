# Phase C — Source–label confounding

| field | source | group0 | group1 | n |
| --- | --- | --- | --- | --- |
| id_city | city1 | 115 | 7 | 122 |
| id_city | city10 | 0 | 5 | 5 |
| id_city | city11 | 0 | 10 | 10 |
| id_city | city12 | 0 | 7 | 7 |
| id_city | city13 | 0 | 7 | 7 |
| id_city | city14 | 0 | 8 | 8 |
| id_city | city15 | 0 | 8 | 8 |
| id_city | city2 | 0 | 2 | 2 |
| id_city | city3 | 0 | 8 | 8 |
| id_city | city4 | 0 | 47 | 47 |
| id_city | city5 | 0 | 8 | 8 |
| id_city | city6 | 0 | 4 | 4 |
| id_city | city7 | 0 | 12 | 12 |
| id_city | city8 | 0 | 2 | 2 |
| id_city | city9 | 0 | 8 | 8 |

City1 có 115 group0/7 group1. Các city khác có tổng 136 người, 0 group0 và 136 group1. Đây là **statistical association**, không phải city gây Dysgraphia.

| field | n | categories | mixed_sources | single_class_sources | label_purity | cramers_v | chi_square | chi_square_asymptotic_p | expected_cells_lt5 | permutation_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| id_city | 258 | 15 | 1 | 14 | 0.972868 | 0.946827 | 231.291987 | 0.000000 | 23 | 0.001000 |
| id_school_ppp | 258 | 35 | 4 | 31 | 0.972868 | 0.957250 | 236.412405 | 0.000000 | 56 | 0.001000 |
| site_type_ppp | 258 | 2 | 1 | 1 | 0.655039 | 0.461384 | 54.921843 | 0.000000 | 0 | 0.001000 |
| date_of_administration | 258 | 7 | 6 | 1 | 0.825581 | 0.671403 | 116.301730 | 0.000000 | 4 | 0.001000 |

Purity = tổng số người thuộc lớp đa số trong từng source / tổng cohort, chỉ là mô tả toàn cohort. Cramér V không bias-corrected. Chi-square dùng correction=False; nhiều expected cells<5 nên p asymptotic chỉ tham khảo. Kiểm chứng bằng999 participant-label permutations, seed42 (p tối thiểu1/1000), không phải phép kiểm định nhân quả và chưa giải quyết clustering/tuyển mẫu.

| field | fold | train_sources | test_sources | unseen_test_sources | test_participants | fraction_test_source_seen |
| --- | --- | --- | --- | --- | --- | --- |
| id_city | 0 | 15 | 11 | 0 | 43 | 1.000000 |
| id_city | 1 | 15 | 11 | 0 | 43 | 1.000000 |
| id_city | 2 | 15 | 9 | 0 | 43 | 1.000000 |
| id_city | 3 | 15 | 11 | 0 | 43 | 1.000000 |
| id_city | 4 | 15 | 10 | 0 | 43 | 1.000000 |
| id_city | 5 | 15 | 12 | 0 | 43 | 1.000000 |
| id_school_ppp | 0 | 35 | 20 | 0 | 43 | 1.000000 |
| id_school_ppp | 1 | 35 | 20 | 0 | 43 | 1.000000 |
| id_school_ppp | 2 | 33 | 21 | 2 | 43 | 0.953488 |
| id_school_ppp | 3 | 34 | 23 | 1 | 43 | 0.976744 |
| id_school_ppp | 4 | 34 | 22 | 1 | 43 | 0.976744 |
| id_school_ppp | 5 | 34 | 25 | 1 | 43 | 0.976744 |
| site_type_ppp | 0 | 2 | 2 | 0 | 43 | 1.000000 |
| site_type_ppp | 1 | 2 | 2 | 0 | 43 | 1.000000 |
| site_type_ppp | 2 | 2 | 2 | 0 | 43 | 1.000000 |
| site_type_ppp | 3 | 2 | 2 | 0 | 43 | 1.000000 |
| site_type_ppp | 4 | 2 | 2 | 0 | 43 | 1.000000 |
| site_type_ppp | 5 | 2 | 2 | 0 | 43 | 1.000000 |
| date_of_administration | 0 | 7 | 6 | 0 | 43 | 1.000000 |
| date_of_administration | 1 | 7 | 7 | 0 | 43 | 1.000000 |
| date_of_administration | 2 | 7 | 7 | 0 | 43 | 1.000000 |
| date_of_administration | 3 | 7 | 6 | 0 | 43 | 1.000000 |
| date_of_administration | 4 | 7 | 5 | 0 | 43 | 1.000000 |
| date_of_administration | 5 | 7 | 6 | 0 | 43 | 1.000000 |

`fold_source_class_distributions.csv` liệt kê source và class counts trong Train/Test của từng fold. Participant độc lập nhưng nguồn có thể đã thấy trong Train. Không gọi đây là source-independent validation.

## Negative control hợp lệ

Mapping nguồn chỉ dùng nhãn Train của năm folds khác. Train prior=(n1+1)/(n+2); source probability=(source_n1+2×Train prior)/(source_n+2). Source chưa thấy fallback Train prior; missing source là category riêng. Strength2 cố định, không tune. Không đưa participant ID/prefix vào diagnostic và không đưa source metadata vào clinical models.

| control | n | accuracy | f1_macro | auc | balanced_accuracy | brier |
| --- | --- | --- | --- | --- | --- | --- |
| id_city | 258 | 0.972868 | 0.972687 | 0.966981 | 0.975524 | 0.032264 |
| id_school_ppp | 258 | 0.957364 | 0.956960 | 0.988568 | 0.958133 | 0.045242 |
| site_type_ppp | 258 | 0.655039 | 0.634613 | 0.668440 | 0.688811 | 0.194830 |
| date_of_administration | 258 | 0.825581 | 0.821503 | 0.820219 | 0.818820 | 0.143447 |
| Train_class_prior | 258 | 0.554264 | 0.356608 | 0.493463 | 0.500000 | 0.247088 |

Diagnostic city đạt accuracy 0.972868 so với prior 0.554264. Đây là khả năng dự đoán **nhãn dataset từ nguồn** dưới CV có source overlap, không phải clinical classifier. Không dùng purity tính toàn cohort để giả làm OOF.

## Identifiability

Không tách được hiệu ứng source và group cho các city chỉ có một lớp. Trong city1 có overlap hai lớp nên có thể mô tả/discriminate trong cùng nguồn, nhưng chỉ 7 positives nên không đủ để khẳng định generalization rộng hay sensitivity chính xác. Nếu giữ city1 làm Test, Train không còn lớp0; nếu giữ city khác làm Test, Test chỉ một lớp. Leave-One-City-Out không giải quyết vấn đề. Site-level có thể có split chứa hai lớp theo đại số nhưng cần xem support và city/protocol vẫn chồng lấp. Bảng đầy đủ `source_validation_feasibility.csv` không đánh đồng khả thi đại số với đủ lực thống kê.
