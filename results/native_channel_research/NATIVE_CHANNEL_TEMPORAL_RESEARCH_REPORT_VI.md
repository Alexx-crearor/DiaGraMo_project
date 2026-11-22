# Phase 7 — Nghiên cứu native-channel temporal

## 1. Tóm tắt điều hành

Benchmark opaque: COMPLETE. Quyết định **B**. Đây là exploratory internal CV trên cohort Slovak, không phải frozen transfer của DiaGraMo.

| Model | Macro-F1 | AUC | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|
| B0 | 0.344262 | 0.479950 | 0.500000 | 0.525000 |
| B1 | 0.598214 | 0.592175 | 0.604845 | 0.600000 |
| B2 | 0.932413 | 0.960457 | 0.929825 | 0.933333 |
| B3 | 0.941337 | 0.969368 | 0.940267 | 0.941667 |
| B4 | 0.941337 | 0.971039 | 0.940267 | 0.941667 |

CNN có F1 nhỉnh hơn thống kê trong lần chạy này, nhưng không hơn đối chứng hoán vị. Không có bằng chứng lợi ích của thứ tự hàng theo phép kiểm tra đã khóa. Điểm cao có thể liên quan đến phân bố giá trị, quan hệ giữa các cột hoặc nguồn thu thập; thí nghiệm này không xác định đóng góp nhân quả của từng yếu tố.

## 2. Dữ liệu gốc và bằng chứng khoa học

[Drotár & Dobeš (2020)](https://doi.org/10.1038/s41598-020-78611-9), Methods, Table 2 và Supplement; [repository gốc](https://github.com/peet292929/Dysgraphia-detection-through-machine-learning/tree/0079fdd34fc5eff0c1c58f9689127b2c73b57f8b). Workbook có 120 người, 57 DYSGR và 63 đối chứng. Nhãn là đánh giá trong nghiên cứu, không tự chuyển thành confirmed clinical diagnosis hay target group của DiaGraMo.

## 3. Kết quả khôi phục schema

Đã đọc lại tài liệu được cache/hash, kiểm tra bảy commit, tree ba tệp và toàn bộ bốn issues. Lịch sử chỉ thay README, workbook hoặc ZIP, không có script acquisition/exporter. Issue 2 không có phản hồi; comment issue 1 cũng chỉ hỏi phần mềm. Không tìm được đúng phần mềm/phiên bản để xác nhận indexed column schema hoặc units. Các giả thuyết c0–c6 giữ mức SUPPORTED bởi pattern, đơn vị và exporter mapping UNVERIFIED. Xem CHANNEL_EVIDENCE_MATRIX.csv, SOURCE_REGISTRY.csv, SOURCE_RECHECK.json và HISTORY_INSPECTION.json.

## 4. Tính toàn vẹn participant và dữ liệu

121 SVC, 120 recording hợp lệ, 2.176.419 samples; một recording rỗng thuộc người vẫn có recording hợp lệ khác. Tái sử dụng đúng participant IDs và năm fold Phase 6, không tạo split mới. Reader giữ nguyên thứ tự, terminal samples và hai sai phân c2 âm. Không sort, sửa, nội suy hay cắt giả thành tám task.

## 5. Confounding của dataset

[Drotár & Dobeš (2020)](https://doi.org/10.1038/s41598-020-78611-9), Methods: case được thu ở Centre for Special-Needs Education, đối chứng tại elementary school. Thiếu site metadata theo người; không tạo site từ label, không source-held-out evaluation và không định lượng tác động nhân quả của nguồn.

## 6. Định nghĩa bài toán opaque-channel

Mỗi recording là ma trận T×7 số native c0–c6. Phân loại workbook label ở cấp participant, không gọi các cột là velocity/pressure/motor angle đã xác minh. Không age, sex hoặc source metadata trong predictors. Không chuyển đơn vị vật lý.

## 7. Các baseline thống kê

B0 tái sử dụng năm checkpoint prior Phase 6, không refit. B1 có sáu thống kê số recording hợp lệ và số mẫu. B2 có 105 features: 77 thống kê marginal và 28 thống kê absolute successive differences. Vì vậy **B2 không hoàn toàn bất biến với thứ tự**. LightGBM cố định 100 trees, 7 leaves, max_depth=3, learning_rate=0,03, min_child_samples=12, reg_lambda=1. StandardScaler chỉ fit Train. Không search hoặc chọn feature bằng Test.

## 8. Kiến trúc CNN compact

13.761 tham số; ba Conv1D song song kernels 5/13/29, mỗi nhánh 32 channels; ReLU, masked pooling, Linear96→32, ReLU, dropout0,1 và Linear32→1. Không attention, SSL, pretrained model hoặc original 14-task head. Window256 không chồng lấp; pooling theo số mẫu hợp lệ rồi trung bình recording. Loss tính theo participant.

## 9. Đối chứng hoán vị thứ tự

B3/B4 cùng kiến trúc, tham số khởi tạo, folds, optimizer, effective batch và 20 epochs. B4 hoán vị đồng bộ bảy cột trong từng window sau normalization; SHA-256 key cố định từ seed42/participant/recording/window-start. Không hoán vị padding, không mix người, không mất mẫu, không chọn permutation theo score. Dù thứ tự hàng bị phá, c2 có thể còn mang dấu vết vị trí; không tuyên bố xóa sạch mọi thông tin vị trí tiềm ẩn.

## 10. Giao thức đã khóa

LOCKED_CONFIG.json và PREEXPERIMENT_DECISIONS.json được ghi trước điểm số mới. Five-fold participant CV giữ nguyên Phase 6, base seed42; code cố định initialization/data-order seed=42+fold cho cả B3/B4, permutation seed42. Đây là một suite với cùng quy tắc seed, không phải chọn seed tốt nhất. Threshold p>0,5, final epoch20, AdamW lr=1e-3/wd=1e-3, clip1, batch8 participant bằng accumulation. Không outer-Test early stopping, threshold tuning, architecture search hoặc seed selection. Quyết định không chạy sensitivity suite base seed43 được khóa trước score. SHA config: `d895cff2f4613c989c660431cde2f85555b133f634ea053c452d7da55ce3e24b`.

## 11. Training và chi phí thực tế

Tổng train wall B3+B4 đã lưu checkpoint: **200.29 giây**; peak tensor allocation: **0.0943 GiB**. Đây không phải total VRAM của driver hay OS peak. Xem RUNTIME_GPU_MEMORY.csv cho từng fold, CPU time, allocated/reserved; RESOURCE_ADMISSION.json cho pilot và dự toán có hệ số 2× trước scoring. POST_TRAIN_PROCESS_MEMORY.json chỉ là RSS quan sát sau training, không phải training peak. Một GPU worker; checkpoint model/optimizer/RNG sau mỗi epoch. Thời gian checkpoint I/O và bootstrap không được gọi là GPU train time.

## 12. Kết quả chính

| Model | Macro-F1 | AUC | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|
| B0 | 0.344262 | 0.479950 | 0.500000 | 0.525000 |
| B1 | 0.598214 | 0.592175 | 0.604845 | 0.600000 |
| B2 | 0.932413 | 0.960457 | 0.929825 | 0.933333 |
| B3 | 0.941337 | 0.969368 | 0.940267 | 0.941667 |
| B4 | 0.941337 | 0.971039 | 0.940267 | 0.941667 |

MODEL_METRICS.csv có toàn bộ endpoint, CI95%, fold mean±SD và confusion counts; FOLD_METRICS.csv giữ từng fold. Không báo partial OOF như full benchmark. Mỗi model hoàn thành có đủ 120 người/5 folds.

## 13. So sánh thứ tự B3 và B4

B3−B4: ΔMacro-F1 = 0.000000; paired CI95% [0.000000, 0.000000]. Corrected=0, harmed=0; số dự đoán nhị phân khác nhau=0. ΔAUC=-0.001671.

CNN có F1 nhỉnh hơn thống kê trong lần chạy này, nhưng không hơn đối chứng hoán vị. Không có bằng chứng lợi ích của thứ tự hàng theo phép kiểm tra đã khóa. Điểm cao có thể liên quan đến phân bố giá trị, quan hệ giữa các cột hoặc nguồn thu thập; thí nghiệm này không xác định đóng góp nhân quả của từng yếu tố. Nếu hard predictions giống nhau, CI bootstrap F1 bằng [0,0] là hệ quả tất định của các dự đoán cố định này; không chứng minh tương đương trên quần thể, seed khác hoặc lần train khác.

## 14. B1, B2 và B3

PAIRED_MODEL_COMPARISON.csv ghi B2−B1 và B3−B2. Chênh lệch CNN−statistical baseline phải đọc cùng CI, không chỉ point estimate. B1 chỉ đo sample length; B2 còn gồm difference summaries. Do capacity, representation và optimization khác nhau, không quy B3−B2 chỉ cho thứ tự. Giữ B3−B4 là controlled comparison chính. B3−B2 ΔMacro-F1=0.008923, paired CI95% [-0.025584, 0.044403]; CI chứa 0 nên chưa chứng minh CNN vượt baseline thống kê.

## 15. Sai số ghép cặp và bất định

2.000 paired IID participant bootstrap, seed42, cùng resampled indices và fixed OOF cho mọi model. CI không bao gồm refitting, seed variance, source clustering, cohort selection hoặc researcher-selection uncertainty. Exploratory multiple comparisons không được hiệu chỉnh. ERROR_OVERLAP.csv có corrected/harmed/both-correct/both-wrong. ERROR_BY_LENGTH.csv dùng quartiles độ dài cố định và Spearman mô tả, không phải causal shortcut test. Không coi 2,17 triệu mẫu là independent participants.

## 16. Giới hạn ngữ nghĩa tín hiệu

Không có physical conversion hoặc motor interpretation. Đơn vị clock, thứ tự/định nghĩa góc, coordinate scale và task boundaries chưa VERIFIED. Phase8 BLOCKED; không hồi diễn giải opaque results thành kinematic biomarkers. Tám điều kiện trong paper không tương đương tám recording/task annotations.

## 17. Liên hệ với mục tiêu DiaGraMo

258 người DiaGraMo và 120 người Slovak khác cohort, target và task/device contract. Không paired comparison, không chung leaderboard. Không load/sửa original Kinematic LightGBM hoặc Stroke Transformer, không chạy Fusion. Đây là kiểm tra phương pháp nội bộ một cohort thứ hai, chưa phải external validation của frozen DiaGraMo.

## 18. Riêng tư và tái lập

Raw, exact IDs/folds, labels từng người, OOF, recording-level arrays và checkpoint chỉ ở data/native_channel_research_local hoặc source local Phase6; các thư mục được Git-ignore. Public chỉ có code, tests, hash và aggregate artifacts. Không gửi email, dùng cloud GPU, paid API, commit/push/merge tự động. REPRODUCE.md mô tả resume và kiểm tra checkpoint.

## 19. Hạn chế khoa học và kiểm thử

N=120, một seed, năm folds tương quan; thiếu source metadata, schema vật lý và task annotation. B4 còn implicit-position cues; B2 không marginal thuần túy. Tests returncode=0; protected files=3266, changed=0; source files=151, changed=0. CPU/CUDA exact optimizer-RNG resume và CPU–GPU tolerance1e-4 được kiểm tra trong tests, không suy thành determinism giữa mọi phiên bản CUDA/hardware.

## 20. Hướng nghiên cứu tiếp theo

**Quyết định B; CONDITIONAL GO cho kiểm chứng phương pháp, NO-GO cho clinical/source-invariant claims.** Ưu tiên duy nhất là replication với metadata nguồn và quy trình tuyển case/control đồng nhất. Nếu nghiên cứu order tiếp, khóa marginal-only và c2/implicit-position ablation trên development cohort độc lập trước khi xem điểm. Không tăng Transformer complexity chỉ từ kết quả hiện tại.
