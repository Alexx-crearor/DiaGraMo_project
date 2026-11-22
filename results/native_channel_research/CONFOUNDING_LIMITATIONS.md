# Giới hạn confounding

[Drotár & Dobeš (2020)](https://doi.org/10.1038/s41598-020-78611-9), Methods: case được thu ở Centre for Special-Needs Education, đối chứng tại elementary school. Thiếu site metadata theo người; không tạo site từ label, không source-held-out evaluation và không định lượng tác động nhân quả của nguồn.

2.000 paired IID participant bootstrap, seed42, cùng resampled indices và fixed OOF cho mọi model. CI không bao gồm refitting, seed variance, source clustering, cohort selection hoặc researcher-selection uncertainty. Exploratory multiple comparisons không được hiệu chỉnh. ERROR_OVERLAP.csv có corrected/harmed/both-correct/both-wrong. ERROR_BY_LENGTH.csv dùng quartiles độ dài cố định và Spearman mô tả, không phải causal shortcut test. Không coi 2,17 triệu mẫu là independent participants.
