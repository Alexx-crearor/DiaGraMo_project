"""UTF-8 Vietnamese research reports, separate from numerical evaluation."""
from .common import md, read, sha, OUT


def write_reports(scores, primary, decision, complete, cost, peak, validation, summary):
    import pandas as pd
    paired = pd.read_csv(OUT/'PAIRED_MODEL_COMPARISON.csv')
    cnn_stats = paired[(paired.left == 'B3') & (paired.right == 'B2')]
    stats_contrast = ''
    if len(cnn_stats):
        row = cnn_stats.iloc[0]
        stats_contrast = (
            f" B3−B2 ΔMacro-F1={row.f1_macro_difference:.6f}, "
            f"paired CI95% [{row.f1_macro_ci95_low:.6f}, {row.f1_macro_ci95_high:.6f}]; "
            "CI chứa 0 nên chưa chứng minh CNN vượt baseline thống kê."
        )
    contrast = (
        f"B3−B4: ΔMacro-F1 = {primary['f1_macro_difference']:.6f}; "
        f"paired CI95% [{primary['f1_macro_ci95_low']:.6f}, "
        f"{primary['f1_macro_ci95_high']:.6f}]. "
        f"Corrected={primary['corrected']}, harmed={primary['harmed']}; "
        f"số dự đoán nhị phân khác nhau={primary['prediction_disagreements']}. "
        f"ΔAUC={primary['auc_difference']:.6f}."
        if primary else "Chưa đủ OOF ghép cặp B3/B4; không tính chênh lệch toàn cohort."
    )
    interpretation = (
        "CNN có F1 nhỉnh hơn thống kê trong lần chạy này, nhưng không hơn đối chứng hoán vị. "
        "Không có bằng chứng lợi ích của thứ tự hàng theo phép kiểm tra đã khóa. "
        "Điểm cao có thể liên quan đến phân bố giá trị, quan hệ giữa các cột hoặc nguồn thu thập; "
        "thí nghiệm này không xác định đóng góp nhân quả của từng yếu tố."
    ) if decision == 'B' else (
        f"Quyết định nghiên cứu {decision}; xem chênh lệch ghép cặp và bất định, "
        "không suy biomarker hoặc tổng quát hóa độc lập nguồn."
    )
    paper = '[Drotár & Dobeš (2020)](https://doi.org/10.1038/s41598-020-78611-9)'
    source = '[repository gốc](https://github.com/peet292929/Dysgraphia-detection-through-machine-learning/tree/0079fdd34fc5eff0c1c58f9689127b2c73b57f8b)'
    sections = [
        ('Tóm tắt điều hành', f'Benchmark opaque: {"COMPLETE" if complete else "PARTIAL"}. Quyết định **{decision}**. Đây là exploratory internal CV trên cohort Slovak, không phải frozen transfer của DiaGraMo.\n\n' + summary + '\n' + interpretation),
        ('Dữ liệu gốc và bằng chứng khoa học', f'{paper}, Methods, Table 2 và Supplement; {source}. Workbook có 120 người, 57 DYSGR và 63 đối chứng. Nhãn là đánh giá trong nghiên cứu, không tự chuyển thành confirmed clinical diagnosis hay target group của DiaGraMo.'),
        ('Kết quả khôi phục schema', 'Đã đọc lại tài liệu được cache/hash, kiểm tra bảy commit, tree ba tệp và toàn bộ bốn issues. Lịch sử chỉ thay README, workbook hoặc ZIP, không có script acquisition/exporter. Issue 2 không có phản hồi; comment issue 1 cũng chỉ hỏi phần mềm. Không tìm được đúng phần mềm/phiên bản để xác nhận indexed column schema hoặc units. Các giả thuyết c0–c6 giữ mức SUPPORTED bởi pattern, đơn vị và exporter mapping UNVERIFIED. Xem CHANNEL_EVIDENCE_MATRIX.csv, SOURCE_REGISTRY.csv, SOURCE_RECHECK.json và HISTORY_INSPECTION.json.'),
        ('Tính toàn vẹn participant và dữ liệu', '121 SVC, 120 recording hợp lệ, 2.176.419 samples; một recording rỗng thuộc người vẫn có recording hợp lệ khác. Tái sử dụng đúng participant IDs và năm fold Phase 6, không tạo split mới. Reader giữ nguyên thứ tự, terminal samples và hai sai phân c2 âm. Không sort, sửa, nội suy hay cắt giả thành tám task.'),
        ('Confounding của dataset', f'{paper}, Methods: case được thu ở Centre for Special-Needs Education, đối chứng tại elementary school. Thiếu site metadata theo người; không tạo site từ label, không source-held-out evaluation và không định lượng tác động nhân quả của nguồn.'),
        ('Định nghĩa bài toán opaque-channel', 'Mỗi recording là ma trận T×7 số native c0–c6. Phân loại workbook label ở cấp participant, không gọi các cột là velocity/pressure/motor angle đã xác minh. Không age, sex hoặc source metadata trong predictors. Không chuyển đơn vị vật lý.'),
        ('Các baseline thống kê', 'B0 tái sử dụng năm checkpoint prior Phase 6, không refit. B1 có sáu thống kê số recording hợp lệ và số mẫu. B2 có 105 features: 77 thống kê marginal và 28 thống kê absolute successive differences. Vì vậy **B2 không hoàn toàn bất biến với thứ tự**. LightGBM cố định 100 trees, 7 leaves, max_depth=3, learning_rate=0,03, min_child_samples=12, reg_lambda=1. StandardScaler chỉ fit Train. Không search hoặc chọn feature bằng Test.'),
        ('Kiến trúc CNN compact', '13.761 tham số; ba Conv1D song song kernels 5/13/29, mỗi nhánh 32 channels; ReLU, masked pooling, Linear96→32, ReLU, dropout0,1 và Linear32→1. Không attention, SSL, pretrained model hoặc original 14-task head. Window256 không chồng lấp; pooling theo số mẫu hợp lệ rồi trung bình recording. Loss tính theo participant.'),
        ('Đối chứng hoán vị thứ tự', 'B3/B4 cùng kiến trúc, tham số khởi tạo, folds, optimizer, effective batch và 20 epochs. B4 hoán vị đồng bộ bảy cột trong từng window sau normalization; SHA-256 key cố định từ seed42/participant/recording/window-start. Không hoán vị padding, không mix người, không mất mẫu, không chọn permutation theo score. Dù thứ tự hàng bị phá, c2 có thể còn mang dấu vết vị trí; không tuyên bố xóa sạch mọi thông tin vị trí tiềm ẩn.'),
        ('Giao thức đã khóa', 'LOCKED_CONFIG.json và PREEXPERIMENT_DECISIONS.json được ghi trước điểm số mới. Five-fold participant CV giữ nguyên Phase 6, base seed42; code cố định initialization/data-order seed=42+fold cho cả B3/B4, permutation seed42. Đây là một suite với cùng quy tắc seed, không phải chọn seed tốt nhất. Threshold p>0,5, final epoch20, AdamW lr=1e-3/wd=1e-3, clip1, batch8 participant bằng accumulation. Không outer-Test early stopping, threshold tuning, architecture search hoặc seed selection. Quyết định không chạy sensitivity suite base seed43 được khóa trước score. SHA config: `' + sha(OUT/'LOCKED_CONFIG.json') + '`.'),
        ('Training và chi phí thực tế', f'Tổng train wall B3+B4 đã lưu checkpoint: **{cost:.2f} giây**; peak tensor allocation: **{peak / 2**30:.4f} GiB**. Đây không phải total VRAM của driver hay OS peak. Xem RUNTIME_GPU_MEMORY.csv cho từng fold, CPU time, allocated/reserved; RESOURCE_ADMISSION.json cho pilot và dự toán có hệ số 2× trước scoring. POST_TRAIN_PROCESS_MEMORY.json chỉ là RSS quan sát sau training, không phải training peak. Một GPU worker; checkpoint model/optimizer/RNG sau mỗi epoch. Thời gian checkpoint I/O và bootstrap không được gọi là GPU train time.'),
        ('Kết quả chính', summary + '\nMODEL_METRICS.csv có toàn bộ endpoint, CI95%, fold mean±SD và confusion counts; FOLD_METRICS.csv giữ từng fold. Không báo partial OOF như full benchmark. Mỗi model hoàn thành có đủ 120 người/5 folds.'),
        ('So sánh thứ tự B3 và B4', contrast + '\n\n' + interpretation + ' Nếu hard predictions giống nhau, CI bootstrap F1 bằng [0,0] là hệ quả tất định của các dự đoán cố định này; không chứng minh tương đương trên quần thể, seed khác hoặc lần train khác.'),
        ('B1, B2 và B3', 'PAIRED_MODEL_COMPARISON.csv ghi B2−B1 và B3−B2. Chênh lệch CNN−statistical baseline phải đọc cùng CI, không chỉ point estimate. B1 chỉ đo sample length; B2 còn gồm difference summaries. Do capacity, representation và optimization khác nhau, không quy B3−B2 chỉ cho thứ tự. Giữ B3−B4 là controlled comparison chính.' + stats_contrast),
        ('Sai số ghép cặp và bất định', '2.000 paired IID participant bootstrap, seed42, cùng resampled indices và fixed OOF cho mọi model. CI không bao gồm refitting, seed variance, source clustering, cohort selection hoặc researcher-selection uncertainty. Exploratory multiple comparisons không được hiệu chỉnh. ERROR_OVERLAP.csv có corrected/harmed/both-correct/both-wrong. ERROR_BY_LENGTH.csv dùng quartiles độ dài cố định và Spearman mô tả, không phải causal shortcut test. Không coi 2,17 triệu mẫu là independent participants.'),
        ('Giới hạn ngữ nghĩa tín hiệu', 'Không có physical conversion hoặc motor interpretation. Đơn vị clock, thứ tự/định nghĩa góc, coordinate scale và task boundaries chưa VERIFIED. Phase8 BLOCKED; không hồi diễn giải opaque results thành kinematic biomarkers. Tám điều kiện trong paper không tương đương tám recording/task annotations.'),
        ('Liên hệ với mục tiêu DiaGraMo', '258 người DiaGraMo và 120 người Slovak khác cohort, target và task/device contract. Không paired comparison, không chung leaderboard. Không load/sửa original Kinematic LightGBM hoặc Stroke Transformer, không chạy Fusion. Đây là kiểm tra phương pháp nội bộ một cohort thứ hai, chưa phải external validation của frozen DiaGraMo.'),
        ('Riêng tư và tái lập', 'Raw, exact IDs/folds, labels từng người, OOF, recording-level arrays và checkpoint chỉ ở data/native_channel_research_local hoặc source local Phase6; các thư mục được Git-ignore. Public chỉ có code, tests, hash và aggregate artifacts. Không gửi email, dùng cloud GPU, paid API, commit/push/merge tự động. REPRODUCE.md mô tả resume và kiểm tra checkpoint.'),
        ('Hạn chế khoa học và kiểm thử', f"N=120, một seed, năm folds tương quan; thiếu source metadata, schema vật lý và task annotation. B4 còn implicit-position cues; B2 không marginal thuần túy. Tests returncode={validation['tests_returncode']}; protected files={validation['protected_files_checked']}, changed={len(validation['protected_files_changed'])}; source files={validation['private_source_files_checked']}, changed={validation['private_source_files_changed']}. CPU/CUDA exact optimizer-RNG resume và CPU–GPU tolerance1e-4 được kiểm tra trong tests, không suy thành determinism giữa mọi phiên bản CUDA/hardware."),
        ('Hướng nghiên cứu tiếp theo', f'**Quyết định {decision}; CONDITIONAL GO cho kiểm chứng phương pháp, NO-GO cho clinical/source-invariant claims.** Ưu tiên duy nhất là replication với metadata nguồn và quy trình tuyển case/control đồng nhất. Nếu nghiên cứu order tiếp, khóa marginal-only và c2/implicit-position ablation trên development cohort độc lập trước khi xem điểm. Không tăng Transformer complexity chỉ từ kết quả hiện tại.'),
    ]
    md('NATIVE_CHANNEL_TEMPORAL_RESEARCH_REPORT_VI.md', '# Phase 7 — Nghiên cứu native-channel temporal\n\n' + '\n\n'.join(f'## {i}. {title}\n\n{text}' for i, (title, text) in enumerate(sections, 1)) + '\n')
    md('EXECUTIVE_SUMMARY_VI.md', '# Tóm tắt Phase 7\n\n' + summary + '\n' + contrast + '\n\n' + interpretation + f'\n\nDecision {decision}; opaque suite complete={complete}.\n')
    md('TEMPORAL_ORDER_ANALYSIS_VI.md', '# Phân tích thứ tự quan sát\n\n' + contrast + '\n\n' + sections[12][1] + '\n\n' + sections[14][1] + '\n')
    md('NEXT_RESEARCH_DECISION_VI.md', '# Quyết định nghiên cứu\n\n' + sections[19][1] + '\n\n' + interpretation + '\n')
    md('CONFOUNDING_LIMITATIONS.md', '# Giới hạn confounding\n\n' + sections[4][1] + '\n\n' + sections[14][1] + '\n')
    md('SEMANTIC_SCHEMA_FEASIBILITY.md', '# Physical semantic study: BLOCKED\n\n' + sections[15][1] + '\n')
    md('BLOCKERS.md', '# Blockers\n\nPhysical schema/units/task boundaries UNVERIFIED. Thiếu source metadata theo participant.\n\n' + ('Opaque suite hoàn tất, không còn fold thiếu.' if complete else 'Opaque suite PARTIAL; xem EXPERIMENT_REGISTRY.csv.') + '\n')
    md('PHASE0_REPOSITORY_CHECK.md', '# Kiểm kê repository\n\nBase c8d81bf, nhánh study/native-channel-temporal-benchmark. Không reset/stash hoặc xóa bốn tệp untracked cũ. Tái sử dụng dataset checksum và năm folds Phase6. Xem HARDWARE_STATUS.json, PROTECTED_INPUT_MANIFEST.json và participant_grouping_validation.json.\n')
    md('SCHEMA_RECOVERY_REPORT_VI.md', '# Khôi phục schema\n\n' + sections[2][1] + '\n\n' + sections[15][1] + '\n\nTìm kiếm được timebox tối đa 60 phút, kết thúc không chờ tác giả; chuyển sang opaque benchmark theo ủy quyền Phase7. Không suy đơn vị từ generic SVC hoặc tài liệu thiết bị chung.\n')
    md('PREPROCESSING_PROTOCOL.md', '# Tiền xử lý native numeric\n\nRead-only Phase6 reader, giữ row order và mọi mẫu hợp lệ. Fit first/second moments chỉ từ Train: equal participants, equal recordings trong participant, equal samples trong recording. Không dùng label trong API preprocessing. StandardScaler B1/B2 fit Train.\n\nNormalize trước fixed permutation. Non-overlap windows256; zero padding sau scaler và explicit valid mask. Không cross-recording; pooling window theo valid sample count, trung bình recordings, loss participant. Giữ terminal samples; không sort c2 hoặc tạo task identities.\n\n' + sections[6][1] + '\n')
    md('LOCKED_PROTOCOL.md', '# Giao thức khóa trước điểm số\n\n' + sections[9][1] + '\n\n' + sections[6][1] + '\n\n' + sections[7][1] + '\n\n' + sections[8][1] + '\n\nResource admission: training-only measured throughput; dự toán đủ cả matched B3+B4. OOM fallback chunk64→32→16→8→CPU, giữ architecture và effective batch8. Checkpoint epoch model/optimizer/RNG; resume không refit completed folds. Văn bản được sửa encoding UTF-8 sau chạy; LOCKED_CONFIG và recipe không thay đổi.\n')
    md('REPRODUCE.md', '# Tái lập và resume\n\nTại project root:\n\n```powershell\n.venv/Scripts/python.exe -X utf8 scripts/native_channel_research/run.py --max-runtime-hours 8\n.venv/Scripts/python.exe -m pytest tests/native_channel_research tests/external_validation -q\n```\n\nMỗi invocation có session budget8h, reserve15 phút. Dùng lại nguồn Phase6 đúng checksum; không tải ZIP lại. Khóa một worker; checkpoint hash/config/split/optimizer/RNG được kiểm tra, completed folds không refit. Test CPU/CUDA cần môi trường hiện tại; không claim portability tuyệt đối. Raw và OOF không public. Đổi config cần protocol/output riêng.\n')
