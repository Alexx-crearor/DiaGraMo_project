"""Build weekly navigation and presentation text from authoritative artifacts."""
import os
from pathlib import Path
import subprocess
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RESEARCH=ROOT/'docs/research'
WEEK=ROOT/'docs/weekly_reports/2026-11-16_to_2026-11-22'
CLEAN=ROOT/'docs/research_cleanup'
LOCK='results/raw_temporal_v2/locked_study/'
NATIVE='results/native_channel_research/'
REPORTS=[
 ('Original reproduction','results/REPORT_VI.md','results/cv/summary.csv','168 historical CV rows; Kinematic/MOMENT/GAF/CWT'),
 ('Previous XAI milestone','results/kinematic_xai/REPORT_VI.md','results/kinematic_xai/exploratory/tables/task_importance.csv','Model explanation; not causal biomarkers'),
 ('Initial raw temporal','results/raw_temporal/REPORT_VI.md','results/raw_temporal/tables/performance_summary.csv','Historical exploratory architectures'),
 ('Recovery development','results/raw_temporal_v2/REVIEW_SNAPSHOT_VI.md','results/raw_temporal_v2/EXPERIMENT_LEDGER.csv','Development/pilot ledger; not whole-cohort OOF'),
 ('SSL recovery','results/raw_temporal_v2/PHASE3_RECOVERY_VI.md','results/raw_temporal_v2/phase3_recovery/status.json','INCOMPLETE; smoke/pilot only'),
 ('Lightweight validation','results/raw_temporal_v2/PHASE4_LITE_REPORT_VI.md','results/raw_temporal_v2/phase4_lite/paired_comparison.csv','Exploratory; distinct from original full nested queue'),
 ('Locked six-fold','results/raw_temporal_v2/locked_study/LOCKED_RESEARCH_REPORT_VI.md',LOCK+'matched_oof_metrics.csv','Complete matched258 OOF; frozen configuration, exploratory cohort'),
 ('Data validity','results/data_audit_v3/COMPREHENSIVE_DATA_AUDIT_VI.md','results/data_audit_v3/metadata_shortcut_metrics.csv','Source–label association; not causal device attribution'),
 ('Provenance','results/provenance_verification/DATASET_PROVENANCE_VERIFICATION_VI.md','results/provenance_verification/LABEL_VERIFICATION_DECISION.json','Published grouping clarified; individual clinical adjudication incomplete'),
 ('External compatibility','results/external_validation/EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md','results/external_validation/TRANSFER_PROTOCOL_DECISION.json','Raw exists; exact frozen transfer invalid under current contract'),
 ('Native-channel benchmark',NATIVE+'NATIVE_CHANNEL_TEMPORAL_RESEARCH_REPORT_VI.md',NATIVE+'MODEL_METRICS.csv','Complete five-fold120; exploratory internal CV, NOT frozen transfer'),
]


def put(folder,name,text):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/name).write_text(text.strip()+'\n',encoding='utf8')


def link(folder,path,label=None):
    assert (ROOT/path).exists() or path.startswith(('docs/research/', 'docs/weekly_reports/')),path
    return f'[{label or path}]({Path(os.path.relpath(ROOT/path,folder)).as_posix()})'


def markdown_table(frame,columns):
    names=list(columns.values());out='| '+' | '.join(names)+' |\n| '+' | '.join(['---']*len(names))+' |\n'
    for _,r in frame.iterrows():
        vals=[]
        for k in columns:
            v=r[k];vals.append(f'{v:.6f}' if isinstance(v,float) else str(v))
        out+='| '+' | '.join(vals)+' |\n'
    return out


def build():
    locked=pd.read_csv(ROOT/(LOCK+'matched_oof_metrics.csv'))
    native=pd.read_csv(ROOT/(NATIVE+'MODEL_METRICS.csv'))
    pairs=pd.read_csv(ROOT/(LOCK+'paired_bootstrap.csv'))
    fusion=pairs[(pairs.model=='Fixed50')&(pairs.metric=='f1_macro')].iloc[0]
    native_pairs=pd.read_csv(ROOT/(NATIVE+'PAIRED_MODEL_COMPARISON.csv'))
    order=native_pairs[(native_pairs.left=='B3')&(native_pairs.right=='B4')].iloc[0]
    stats=native_pairs[(native_pairs.left=='B3')&(native_pairs.right=='B2')].iloc[0]
    main_table=markdown_table(locked,{'model':'Model','f1_macro':'Macro-F1 OOF','accuracy':'Accuracy','auc':'AUC'})
    native_table=markdown_table(native,{'model':'Model','f1_macro':'Macro-F1 OOF','accuracy':'Accuracy','auc':'AUC','balanced_accuracy':'Balanced Accuracy'})
    reportlinks='\n'.join(f'- {stage}: {link(RESEARCH,report)} — {note}.' for stage,report,_,note in REPORTS)
    put(RESEARCH,'README.md',f'''# Bản đồ nghiên cứu DiaGraMo

Đọc theo thứ tự: mốc XAI trước → Temporal/locked OOF → validity/provenance → independent opaque benchmark → quyết định tuần tới.

- [Tiến trình theo Git](RESEARCH_TIMELINE.md).
- [Ba kiến trúc và phạm vi khác nhau](ARCHITECTURE_EVOLUTION.md).
- [Registry có nguồn metric và commit](EXPERIMENT_REGISTRY.csv).
- [Mục lục report và bảng gốc](RESULTS_INDEX.md).
- [Giới hạn khoa học](VALIDITY_AND_LIMITATIONS.md).
- {link(RESEARCH,'docs/weekly_reports/2026-11-16_to_2026-11-22/WEEKLY_PROGRESS_VI.md','Báo cáo tuần')}.

## Điểm vào theo giai đoạn

{reportlinks}

`src/diagramo/branches/` và pipeline baseline vẫn KEEP_ACTIVE. Code Temporal, audit, provenance, external/native cũng được giữ tại đường dẫn cũ để tái lập; chỉ chạy khi có protocol mới được chủ động lựa chọn. Các report/metric/checkpoint hoàn tất là KEEP_HISTORICAL, không được ghi đè. REVIEW_FOR_REMOVAL là đề xuất cần chủ sở hữu duyệt, không phải danh sách đã xóa.
''')
    index='| Giai đoạn | Báo cáo | Nguồn định lượng/trạng thái | Phạm vi |\n|---|---|---|---|\n'
    for stage,report,source,note in REPORTS:
        index+=f'| {stage} | {link(RESEARCH,report,"Report")} | {link(RESEARCH,source,"Artifact gốc")} | {note} |\n'
    put(RESEARCH,'RESULTS_INDEX.md','# Mục lục kết quả và nguồn định lượng\n\n'+index+'\nKhông dùng bảng development, partial folds hoặc cohort Slovak như kết quả confirmatory của DiaGraMo. Individual OOF/SHAP rows và hình local-case không thuộc bộ trình chiếu công khai.\n')
    log=subprocess.check_output(['git','log','--reverse','--format=%h|%aI|%s','80f9493..abee084'],cwd=ROOT,text=True,encoding='utf8')
    timeline='| Commit | Author timestamp có múi giờ | Thay đổi theo Git |\n|---|---|---|\n'
    for line in log.splitlines():
        sha,date,message=line.split('|',2)
        timeline+=f'| `{sha}` | {date} | {message} |\n'
    put(RESEARCH,'RESEARCH_TIMELINE.md',f'''# Tiến trình sau mốc Kinematic XAI

Khoảng báo cáo theo yêu cầu: 16–22/11/2026. XAI ở commit `80f9493` (16/11 09:29 +07) được xem là **mốc đã báo cáo**, dù nằm cùng ngày đầu cửa sổ tuần; không tính lại thành kết quả mới. Ngày dưới đây là author timestamp từ Git, không tự coi ngày commit là thời điểm hoàn tất mọi fit.

{timeline}

## Tiến trình khoa học được chứng minh

1. XAI giải thích mô hình Kinematic, `copy` chiếm khoảng35,34% absolute SHAP mass; không chứng minh tác vụ này là biomarker.
2. Task-feature Transformer đã được thử rồi bỏ khỏi active tree ngày16/11; không tái đưa lại kiến trúc đó. Chuyển sang raw temporal để kiểm tra thông tin bị nén bởi thống kê.
3. Initial raw temporal và recovery Phase0–2 cung cấp development/synthetic evidence; không gọi các probe là full CV.
4. SSL gặp giới hạn kỹ thuật/compute; deterministic interpolation được sửa, smoke và pilot hoàn tất nhưng full SSL vẫn INCOMPLETE. Không có kết quả embedding cải thiện để báo cáo.
5. Lightweight validation và locked study ngày21/11 lưu đủ sáu folds. Stroke Temporal thấp hơn Kinematic; Fixed50 không chứng minh cải thiện.
6. Audit V3 (`f419530`) phát hiện source–label association mạnh; provenance (`e26b0a1`) làm rõ target group rộng hơn dysgraphia flag.
7. External audit (`c8d81bf`) xác minh Slovak có raw thật nhưng không đáp ứng frozen14task/channel contract.
8. Native benchmark (`abee084`) chạy đủ B0–B4; B3/B4 cùng binary predictions. Đây là cohort-internal exploratory validation mới, không frozen external transfer.

Nguồn chi tiết và metric ở [Results index](RESULTS_INDEX.md), [Registry](EXPERIMENT_REGISTRY.csv). Không có thí nghiệm mới trong cleanup này.
''')
    put(RESEARCH,'ARCHITECTURE_EVOLUTION.md','''# Ba tuyến kiến trúc cần phân biệt

## Mốc trước: Kinematic và XAI

```mermaid
flowchart LR
 A[14 tác vụ viết] --> B[74 thống kê mỗi tác vụ]
 B --> C[1036 features]
 C --> D[LightGBM]
 D --> E[TreeSHAP và task/group analysis]
```

Code: `src/diagramo/features.py`, `branches/kinematic.py`, `kinematic_xai.py`. XAI diễn giải mô hình, không nhận diện nguyên nhân bệnh.

## Raw Temporal đã triển khai

```mermaid
flowchart LR
 A[SVC bảy tín hiệu gốc] --> B[21 processed channels]
 B --> C[Non-overlap windows tối đa128]
 C --> D[MultiScaleLocal kernels5/13/29]
 D --> E[Bốn local tokens và segment aggregation]
 E --> F[RelativeStrokeBlock hai heads]
 F --> G[Task identity14 và learned participant pooling]
 G --> H[Temporal probability pT]
 K[Kinematic LightGBM pK] --> J[Fixed50: 0.5pK + 0.5pT]
 H --> J
```

Nguồn: `src/diagramo/temporal/preprocessing.py`, `dataset.py`, `segmentation.py`, `src/diagramo/temporal_v2/multiscale.py`, config `configs/raw_temporal_v2/phase4_lite.json`. Khoảng25.252 tham số. Relative attention hoạt động giữa segment/stroke representations trong task groups; task identity embedding và pooling **không phải task-level Transformer attention**. Giữ thứ tự vật lý và mask theo code; không diễn giải attention như causal attribution.

## Slovak opaque benchmark độc lập

```mermaid
flowchart LR
 A[SVC numeric c0 đến c6] --> B[Train-only participant-balanced normalization]
 A --> S[Length và statistical LightGBM]
 B --> C[Windows256 giữ đầy đủ mẫu]
 C --> D[Ordered compact CNN B3]
 C --> E[Fixed synchronized row permutation]
 E --> F[Cùng compact CNN B4]
 D --> G[Participant OOF]
 F --> G
```

Nguồn: `scripts/native_channel_research/compact_multiscale_cnn.py`, `participant_dataset.py`, `sequence_permutation.py`. 13.761 tham số, không relative attention, task identities hoặc14task head. Đây không phải cùng Stroke Temporal25.252 tham số. Không chuyển tên numeric columns thành pressure/timestamp đã VERIFIED; B4 có thể giữ implicit positional cues qua giá trị cột.
''')
    put(RESEARCH,'VALIDITY_AND_LIMITATIONS.md','''# Giới hạn bắt buộc đi kèm kết quả

- Target DiaGraMo là `group`, một handwriting-disability grouping rộng. Không đồng nhất group1 với dysgraphia-specific flag hoặc chẩn đoán được xác nhận độc lập từng người; ongoing assessment nằm trong mô tả nguồn.
- Cohort258 giữ nguyên115/143, năm disagreement với specific flag và ba omission flags. Không đổi label/cohort trong cleanup.
- City–label association mạnh không tự chứng minh ID trực tiếp vào features hoặc thiết bị gây ra kết quả. Source effects và motor effects chưa identifiable từ dữ liệu hiện có.
- Participant-independent CV không thành source-independent CV. Historical researcher selection vẫn giới hạn các study đã inspected cohort.
- Attribution, attention, subgroup analysis và OOF bootstrap không xác lập cơ chế bệnh hay causal biomarker.
- Locked six folds hoàn tất; original full nested Phase4/SSL không được coi hoàn tất chỉ vì Lite xong. Bootstrap giữ mô hình cố định, không bao gồm toàn bộ training/selection/source uncertainty.
- Slovak120 là cohort/target/task khác. B0–B4 là internal exploratory CV tại cohort đó, không frozen external validation của DiaGraMo.
- B2 có28 order-dependent difference summaries; B4 shuffle vẫn có thể giữ vị trí ngầm. B3/B4 hard-prediction equality không chứng minh mọi temporal information vô ích.
- Không dùng individual clinical/SHAP/OOF hoặc raw trace hình nhận dạng được trong slide công khai. Hình aggregate cần giữ chú thích scope/uncertainty; audit nguồn15cities có nhóm nhỏ, trình bày nội bộ ở dạng đã gộp là ưu tiên.
''')
    weekly=f'''# Báo cáo tiến độ nghiên cứu — 16–22/11/2026

## 1. Công việc đã hoàn thành trước mốc báo cáo này

Kinematic Task-Level XAI (`80f9493`) là điểm xuất phát đã báo cáo: TreeSHAP, task/group importance, grouped permutation, LOTO, fold stability và Train-only importance. Baseline Kinematic Macro-F1≈0,8472; `copy` đứng đầu SHAP task mass ở sáu folds. Đây là hành vi mô hình, không causal clinical biomarker và không đủ để kết luận số task tối thiểu. XAI gợi câu hỏi: thống kê74features/task có bỏ lỡ diễn biến trong mẫu viết không?

Nguồn: {link(WEEK,'results/kinematic_xai/REPORT_VI.md','XAI report')} và {link(WEEK,'results/kinematic_xai/exploratory/tables/task_importance.csv','task importance')}.

## 2. Mục tiêu nghiên cứu trong tuần

Kiểm tra raw chronological representations; phát triển Stroke-level relative attention và thử fixed fusion với Kinematic. Sau khi kết quả không vượt baseline, tập trung kiểm toán target/nguồn thay vì tăng capacity. Tìm cohort độc lập để kiểm tra giới hạn phương pháp; không mặc định external data tương thích frozen model.

## 3. Kiến trúc Temporal thực tế

Bảy tín hiệu SVC →21 processed channels →non-overlap windows≤128 →multi-scale CNN kernels5/13/29 →bốn local tokens/window →segment aggregation →RelativeStrokeBlock hai heads →task identity embeddings14 →learned participant pooling →pT. Khoảng25.252 tham số. Relative attention ở segment/stroke level; task pooling không phải task-level Transformer. Kinematic xuất pK; fusion cố định pF=0,5pK+0,5pT.

Nguồn code: {link(WEEK,'src/diagramo/temporal_v2/multiscale.py','MultiScaleLocal/RelativeStrokeBlock/MultiScaleModel')}; {link(WEEK,'docs/research/ARCHITECTURE_EVOLUTION.md','sơ đồ và phân biệt ba tuyến')}.

## 4. Kết quả chính trên DiaGraMo

{main_table}

Đủ258 người/sáu folds, participant stratification seed42; cùng người/folds khi so sánh. Fixed50−Kinematic ΔMacro-F1={fusion.delta:.6f}, paired CI95%[{fusion.lower95:.6f},{fusion.upper95:.6f}], gồm0. Fusion tăng ở3/6folds, giảm3/6; corrected20/harmed22. Temporal thấp hơn Kinematic; fixed fusion không chứng minh cải thiện. Không diễn giải SD như CI. Fold mean±SD ở bảng gốc; paired bootstrap2000 lần trong lớp, giữ fitted models cố định, không sửa source/researcher-selection bias.

Nguồn: {link(WEEK,LOCK+'matched_oof_metrics.csv','OOF metrics')}, {link(WEEK,LOCK+'fold_mean_sd.csv','fold mean±SD')}, {link(WEEK,LOCK+'paired_bootstrap.csv','paired CI')}.

## 5. Data validity và source confounding

| Nguồn đã gộp | group0 | group1 | Tổng |
|---|---:|---:|---:|
| City1 |115|7|122|
| Các city khác |0|136|136|
| Tổng |115|143|258|

15city categories; Cramér’s V≈0,946827. City-only Train-mapping control đạt OOF Accuracy≈0,972868, cao hơn class prior≈0,554264. Đây là khả năng dự đoán dataset target từ nguồn, không clinical accuracy. Participant-independent CV vẫn trộn nguồn giữa Train/Test. Chưa chứng minh direct ID leakage hoặc causal device effect; thiếu overlap hai lớp tại nhiều nguồn làm việc tách motor/source effects chưa khả thi.

Nguồn: {link(WEEK,'results/data_audit_v3/source_label_contingency.csv','contingency')}, {link(WEEK,'results/data_audit_v3/metadata_shortcut_metrics.csv','negative controls')}, {link(WEEK,'results/data_audit_v3/source_association_metrics.csv','association statistics')}.

## 6. Dataset provenance và clinical label

Đã nối data descriptor và Zenodo release trong Phase5. `group` là handwriting-disability grouping rộng; dysgraphia-specific flag và ongoing clinical/educational assessment không phải cùng biến. Năm disagreement quan sát được, một phần tương thích mô tả đang assessment, nhưng individual clinical adjudication, một số rules CSV/RDS→merged và duplicate handling chưa có xác minh độc lập. Không đổi nhãn hay loại omission cases sau khi xem kết quả.

Nguồn: {link(WEEK,'results/provenance_verification/DATASET_PROVENANCE_VERIFICATION_VI.md','provenance report')}.

## 7. Khảo sát cohort Slovak độc lập

Drotár & Dobeš2020:120 người,57DYSGR/63controls;121SVC,120recording hợp lệ,2.176.419raw samples. Một file rỗng nhưng người tương ứng còn recording hợp lệ. Bảy numeric columns không header, units/exporter order chưa VERIFIED; tám điều kiện trong paper không có raw task cutpoints và không khớp14canonical tasks. Frozen DiaGraMo transfer hiện không hợp lệ theo contract. Recruitment case/control cũng đi qua institutional pathways khác nhau; không có site metadata cá nhân để source-held-out validation.

Nguồn: {link(WEEK,'results/external_validation/EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md','external compatibility report')}.

## 8. Native-channel benchmark B0–B4

{native_table}

B0 reuse prior; B1 sample-length LightGBM; B2 opaque statistical LightGBM; B3 ordered compact CNN; B4 cùng CNN với fixed synchronized valid-row permutation. Giữ đúng120 người/five participant folds Phase6, base seed42, no outer-Test selection; 20epochs CNN cố định. B3 và B4 có cùng120binary predictions: ΔF1={order.f1_macro_difference:.6f}, corrected={int(order.corrected)}, harmed={int(order.harmed)}. ΔAUC={order.auc_difference:.6f}, CI95%[{order.auc_ci95_low:.6f},{order.auc_ci95_high:.6f}]. CI F1=[0,0] suy biến do fixed hard-prediction equality, không population equivalence. B3−B2 ΔF1={stats.f1_macro_difference:.6f}, CI95%[{stats.f1_macro_ci95_low:.6f},{stats.f1_macro_ci95_high:.6f}], chưa chứng minh CNN vượt statistical baseline.

Macro-F1≈94,13% không phải successful DiaGraMo external transfer. Một main seed; physical schema chưa rõ; B2 có28difference summaries phụ thuộc thứ tự; B4 có thể giữ implicit-position cues. Chưa có evidence disease-specific temporal mechanism.

Nguồn: {link(WEEK,NATIVE+'MODEL_METRICS.csv','metrics')}, {link(WEEK,NATIVE+'PAIRED_MODEL_COMPARISON.csv','paired contrasts')}, {link(WEEK,NATIVE+'NATIVE_CHANNEL_TEMPORAL_RESEARCH_REPORT_VI.md','report')}.

## 9. Bài học khoa học

- Tích cực: hoàn thành full matched six-fold evaluation, audit dữ liệu/clinical provenance và benchmark cohort thứ hai có permutation control.
- Âm tính: Temporal không vượt Kinematic, fixed fusion không cải thiện đáng tin; ordered CNN không có ưu thế phân loại so với permuted control.
- Chưa kết luận: CNN hơn statistics, motor-vs-source contribution, physical channel meanings và clinical/source-independent validity.
- SSL chỉ smoke/pilot, trạng thái **INCOMPLETE**; original expensive nested Phase4 không được chạy trong tuần cleanup.
- Bước cleanup này chỉ đọc/kiểm thử/tổ chức lại; không retrain, tối ưu hoặc tạo metric thí nghiệm mới.

## 10. Kế hoạch tuần tới

Ưu tiên replication/source-aware data design và khóa shortcut/implicit-position hypothesis trước score. Development-only ablation marginal-only, channel/c2 removal và order robustness là lựa chọn nhẹ sau khi protocol được duyệt; chưa chạy trong task này. Chưa ưu tiên Transformer lớn hơn. Chi tiết ở [NEXT_WEEK_PLAN_VI.md](NEXT_WEEK_PLAN_VI.md).
'''
    put(WEEK,'WEEKLY_PROGRESS_VI.md',weekly)
    put(WEEK,'NEXT_WEEK_PLAN_VI.md','''# Kế hoạch tuần tới — theo giá trị khoa học và tính khả thi

| Ưu tiên | Công việc đề xuất | Câu hỏi / tiêu chí hoàn thành | Giới hạn |
|---|---|---|---|
| P0 | Source-aware replication/data design | Có cả hai lớp ở nhiều nguồn, metadata acquisition và prospective held-out source | Cần owner/đối tác dữ liệu; chưa có quyền hoặc nguồn mới tự động |
| P1 | Khóa shortcut hypothesis trên development cohort | Marginal-only baseline riêng; c2/channel ablation; order controls không để giá trị cột tiết lộ vị trí | Không tune lại outer OOF đã inspected; chưa chạy |
| P2 | Kiểm tra robustness/calibration có cross-fitting | Phân biệt capacity, calibration và sequential information theo budget cố định | Protocol mới, Train-only mọi fit; chưa thực hiện |
| P3 | Làm rõ exporter/clinical codebook và release permissions | Nguồn primary về units/task cutpoints/clinical rules | Không gửi liên hệ tự động; không chặn nghiên cứu opaque hợp lệ |

Quyết định: CONDITIONAL GO cho kiểm chứng giả thuyết có kiểm soát; NO-GO cho clinical biomarker/source-invariance claims. Chưa có lý do đủ mạnh để ưu tiên Transformer lớn hơn, full SSL hay full nested architecture search. Điểm94% trên Slovak không giải quyết source confounding.
''')
    slides=[
      ('Điểm xuất phát: XAI đã báo cáo',['Baseline Kinematic F1≈0,8472.','TreeSHAP/task/group/permutation/LOTO đã hoàn tất trước mốc này.','copy nổi bật nhưng attribution không phải causal biomarker.'],'results/kinematic_xai/figures/task_importance.png','results/kinematic_xai/exploratory/tables/task_importance.csv','XAI đặt câu hỏi mới, không phải kết quả mới của tuần.'),
      ('Vì sao thử raw temporal?',['74features/task nén một chuỗi dài.','Giả thuyết: chronological structure có thể bổ sung thông tin.','Không mặc định mô hình lớn sẽ tốt hơn.'],'docs/research/ARCHITECTURE_EVOLUTION.md','src/diagramo/temporal/preprocessing.py','Động cơ là kiểm chứng biểu diễn, chưa phải chứng minh temporal biomarker.'),
      ('Kiến trúc đã triển khai',['7raw→21processed channels; windows≤128.','Multi-scale CNN5/13/29, segment-relative attention.','14task identities và learned participant pooling; khoảng25.252parameters.','Task pooling không phải task-level Transformer.'],'results/raw_temporal_v2/figures/p2_architecture.png','src/diagramo/temporal_v2/multiscale.py','Chỉ rõ nhánh M3 trong hình; tách attention segment khỏi task aggregation.'),
      ('Giao thức đánh giá và trạng thái',['258participants;6folds seed42 giữ nguyên.','Frozen final configuration, Train-only preprocessing.','Full sixfold OOF hoàn tất; cohort đã inspected nên exploratory.','SSL chỉ smoke/pilot; full original nested queue chưa hoàn tất.'],'docs/research/EXPERIMENT_REGISTRY.csv','results/raw_temporal_v2/locked_study/matched_fold_metrics.csv','Completed fits khác với completed original research programme.'),
      ('Kết quả bất ngờ: Fusion không vượt baseline',['Kinematic0,8472;Temporal0,7132;Fixed500,8407 F1.','ΔFusion−Kinematic≈−0,00648, CI chứa0.','Fusion corrected20/harmed22, không chỉ nhìn fold tốt.'],'results/raw_temporal_v2/locked_study/matched_oof_metrics.csv','results/raw_temporal_v2/locked_study/paired_bootstrap.csv','Không có reliable fusion improvement.'),
      ('Source–label confounding',['City1:115group0/7group1; còn lại0/136.','City-only OOF Accuracy≈97,29%; V≈0,947.','Không đồng nhất source association với direct leakage hoặc causal device effect.'],'results/data_audit_v3/figures/source_label_composition.png','results/data_audit_v3/metadata_shortcut_metrics.csv','Participant-independent chưa đủ cho source-independent validity.'),
      ('Clinical target thực sự là gì?',['group là broader handwriting-disability grouping.','Specific dysgraphia flag và ongoing assessment cần phân biệt.','5disagreements,3omission cases; không tự sửa label/cohort.'],'docs/weekly_reports/2026-11-16_to_2026-11-22/WEEKLY_PROGRESS_VI.md','results/provenance_verification/LABEL_VERIFICATION_DECISION.json','Không diễn đạt mọi group1 là confirmed dysgraphia.'),
      ('Cohort độc lập: có raw nhưng không tương thích frozen model',['Slovak120người,57/63labels,120valid recordings.','7numeric columns không header, units/taskcutpoints thiếu.','8documented conditions khác14canonical tasks.','Không load frozen DiaGraMo để giả external validation.'],'results/external_validation/CHANNEL_COMPATIBILITY_MATRIX.csv','results/external_validation/TRANSFER_PROTOCOL_DECISION.json','Compatibility audit là kết quả có ích dù transfer chưa hợp lệ.'),
      ('Opaque benchmark B0–B4',['Prior0,3443;Length0,5982;Statistics0,9324 F1.','Ordered CNN và Permuted CNN đều0,9413.','5participant folds reused; cùng architecture/epochs của B3/B4.','Không dùng score để đổi threshold/model.'],'results/native_channel_research/MODEL_METRICS.csv','results/native_channel_research/LOCKED_CONFIG.json','Điểm cao là cohort-internal exploratory result.'),
      ('Order control và kết luận thận trọng',['B3/B4 cùng binary predictions trên120người.','ΔF1=0; ΔAUC CI chứa0.','B3−B2 F1 CI cũng chứa0.','B2 có difference features; B4 có thể giữ vị trí ngầm.'],'results/native_channel_research/PAIRED_MODEL_COMPARISON.csv','results/native_channel_research/ERROR_OVERLAP.csv','Chưa chứng minh lợi ích của order hay temporal biomarker.'),
      ('Bài học và kế hoạch tiếp theo',['Giữ Kinematic làm baseline; không tăng architecture vô điều kiện.','Ưu tiên source-aware replication và clinical/provenance constraints.','Khóa marginal-only/channel/order hypotheses trên development data mới.','Repository có registry, manifest, tests và report navigation; không xóa lịch sử.'],'docs/weekly_reports/2026-11-16_to_2026-11-22/NEXT_WEEK_PLAN_VI.md','docs/research/VALIDITY_AND_LIMITATIONS.md','CONDITIONAL GO cho nghiên cứu, NO-GO cho clinical/source-invariant claims.'),
    ]
    presentation='# Nội dung trình bày — 11 slides, chưa tạo PowerPoint\n\nMỗi slide giữ riêng implementation, evaluation và inference. XAI là mốc trước; không gộp hai cohort vào một leaderboard.\n'
    for i,(title,points,figure,artifact,takeaway) in enumerate(slides,1):
        presentation+=f'\n## Slide {i}. {title}\n\n'+'\n'.join('- '+p for p in points)+f'\n\nHình/bảng: {link(WEEK,figure)}. Nguồn chính xác: {link(WEEK,artifact)}.\n\n**Thông điệp:** {takeaway}\n\nGhi chú nói: giữ chú thích scope/uncertainty; không trình chiếu individual OOF/clinical rows.\n'
    put(WEEK,'PRESENTATION_CONTENT_VI.md',presentation)
    figures=[
      ('XAI task mass','results/kinematic_xai/figures/task_importance.png','Previous XAI','copy dominates fitted model attribution','READY','No direct ID; aggregate across258',1),
      ('XAI→Temporal transition','docs/research/ARCHITECTURE_EVOLUTION.md','Cross-stage','Three distinct pipelines','READY (Mermaid; slide redraw optional)','No individual data',2),
      ('Stroke architecture','results/raw_temporal_v2/figures/p2_architecture.png','Recovery/locked','M3 relative stroke attention; task pooling','READY; highlight M3','No individual data',3),
      ('Matched Kinematic/Temporal/Fixed50','results/raw_temporal_v2/locked_study/matched_oof_metrics.csv','Locked258','Temporal/fusion do not beat baseline','READY table','Aggregate only',5),
      ('Fusion paired uncertainty','results/raw_temporal_v2/locked_study/paired_bootstrap.csv','Locked258','Delta CI contains zero','READY table','Aggregate only',5),
      ('Source-label composition','results/data_audit_v3/figures/source_label_composition.png','AuditV3','Strong source association, not causality','READY internal; prefer City1/other grouping for wider release','No IDs; rare city counts need disclosure review',6),
      ('Dataset compatibility','results/external_validation/CHANNEL_COMPATIBILITY_MATRIX.csv','External','Raw availability does not imply frozen contract match','READY table','Aggregate only',8),
      ('Slovak B0–B4','results/native_channel_research/MODEL_METRICS.csv','Native120','High internal scores; distinct cohort','READY table','Aggregate only',9),
      ('B3−B4 paired control','results/native_channel_research/PAIRED_MODEL_COMPARISON.csv','Native120','Same hard predictions; AUC difference uncertain','READY table','Aggregate only',10),
      ('Research roadmap','docs/weekly_reports/2026-11-16_to_2026-11-22/NEXT_WEEK_PLAN_VI.md','Planning','Source-aware replication before capacity','READY table','No individual data',11),
      ('Participant local SHAP/waterfalls','results/kinematic_xai/figures/waterfall_false_negative.png','Previous XAI','Individual explanation not required for weekly narrative','NOT RECOMMENDED; privacy review required','Individual attribution/clinical prediction even if ID absent','Backup only'),
    ]
    figtext='# Figure/table index — giữ nguyên ảnh cũ\n\nĐã xem trực quan task_importance.png, p2_architecture.png và source_label_composition.png; đối chiếu kiến trúc với code. Không render lại ảnh lịch sử. Các bảng native/locked được đọc từ CSV gốc. READY là đủ dùng nội dung, không phải chứng nhận quyền phát hành lâm sàng.\n\n| Figure/table | Source | Stage | Message | Status | Privacy | Slide |\n|---|---|---|---|---|---|---|\n'
    for title,path,stage,message,status,privacy,slide in figures:
        figtext+=f'| {title} | {link(WEEK,path)} | {stage} | {message} | {status} | {privacy} | {slide} |\n'
    figtext+='\nKhông đưa participant_sensor_signatures, embeddings scatter có individual points, raw writing traces, local SHAP cases hoặc participant_complementarity rows vào deck mặc định. B0–B4 và matched results dùng bảng có sẵn nên không cần tạo PNG mới.\n'
    put(WEEK,'FIGURE_INDEX.md',figtext)
    claims=[
      ('XAI explains fitted model, not causality','results/kinematic_xai/exploratory/tables/task_importance.csv','DiaGraMo258','copy share≈35.34%,rank1 sixfolds','Held-out sixfold attribution','Strong descriptive model evidence','Selection/source confounding; SHAP not group causal effect','copy có đóng góp lớn cho mô hình đã fit','copy là biomarker hoặc đủ chẩn đoán'),
      ('Temporal uses segment attention and task pooling','src/diagramo/temporal_v2/multiscale.py','DiaGraMo258','MultiScaleLocal/RelativeStrokeBlock/task_id/Pool','Code inspection and tests','Verified implementation','Not task-level Transformer attention','Relative stroke attention và learned participant pooling','Transformer giữa14tasks đã được xác nhận'),
      ('Locked sixfold OOF complete',LOCK+'matched_fold_metrics.csv','DiaGraMo258','3models×6folds;258matched participants', 'Fixed configuration sixfold participant CV','Verified coverage','Exploratory cohort/history','Đủ sáu folds matched OOF','Full original nested search/SSL đã hoàn tất'),
      ('Fixed50 lower pooled F1 and CI crosses zero',LOCK+'paired_bootstrap.csv','DiaGraMo258',f'delta={fusion.delta};CI=[{fusion.lower95},{fusion.upper95}]','2000class-stratified paired fixed-OOF bootstrap','Evidence against reliable improvement','Not refit/selection/source uncertainty','Fusion chưa chứng minh cải thiện Kinematic','Fusion improved diagnosis'),
      ('Source metadata predicts target strongly','results/data_audit_v3/metadata_shortcut_metrics.csv','DiaGraMo258','City-only Accuracy0.972868','Train-only category mapping; held-out participant folds','Strong association diagnostic','Not causal or source-held-out','Source–label association rất mạnh','ID leakage/thiết bị gây ra kết quả đã được chứng minh'),
      ('Clinical group broader than dysgraphia flag','results/provenance_verification/LABEL_VERIFICATION_DECISION.json','DiaGraMo258','Published grouping/ongoing assessment; individual adjudication incomplete','Source document/codebook audit','Definition clarified; individual status incomplete','Some raw clinical/source processing unavailable','group là broader handwriting-disability category','Mọi group1 confirmed dysgraphia độc lập'),
      ('Slovak benchmark is not frozen transfer','results/external_validation/TRANSFER_PROTOCOL_DECISION.json','Slovak120','7opaque columns;8conditions vs14tasks','Compatibility contract audit','Verified incompatibility/remaining semantic gaps','No exact exporter units/taskboundaries','Exploratory internal CV tại cohort thứ hai','DiaGraMo frozen model external accuracy94%'),
      ('B3 and B4 hard predictions identical',NATIVE+'PAIRED_MODEL_COMPARISON.csv','Slovak120',f'disagreements={order.prediction_disagreements};corrected={order.corrected};harmed={order.harmed}','Matched120participant OOF, fixed threshold','Verified within completed suite','One main seed; implicit positional cues remain','Không thấy lợi ích order trong protocol này','Thời gian vô ích cho dysgraphia nói chung'),
      ('B3 versus B2 F1 CI includes zero',NATIVE+'PAIRED_MODEL_COMPARISON.csv','Slovak120',f'delta={stats.f1_macro_difference};CI=[{stats.f1_macro_ci95_low},{stats.f1_macro_ci95_high}]','2000IID paired fixed-OOF bootstrap','Inconclusive superiority','B2includes differences; no training/site uncertainty','Chưa chứng minh CNN vượt statistical baseline','CNN cải thiện có ý nghĩa đã xác nhận'),
      ('Source and motor causal effects not identifiable','results/provenance_verification/DATASET_PROVENANCE_VERIFICATION_VI.md','Both cohorts','Confounded recruitment, insufficient source-label overlap','Audit and source evidence','Unresolved identification','No prospective/source-balanced cohort','Chưa tách được motor/source contribution','Đã đo phần trăm causal contribution từ thiết bị'),
      ('Presentation assets are aggregate-safe selections','docs/weekly_reports/2026-11-16_to_2026-11-22/FIGURE_INDEX.md','Both cohorts','Aggregate plots/tables selected; local cases excluded','Visual inspection and artifact review','Bounded privacy review','Rare subgroup counts need owner review for wider release','Hình tổng hợp phục vụ seminar nội bộ','Mọi ảnh trong repo đều an toàn public'),
      ('SSL is pilot-only','results/raw_temporal_v2/phase3_recovery/status.json','Development Train','phase3_complete=false','CUDA smoke and bounded one-epoch subset pilot','Verified incomplete status','No downstream full SSL suite','Đã xác minh kỹ thuật; full SSL chưa chạy','SSL đã cải thiện embedding toàn cohort'),
    ]
    claimrows=[]
    for claim,source,dataset,evidence,protocol,strength,limits,allowed,avoid in claims:
        commit=subprocess.check_output(['git','log','-1','--format=%H','--',source],cwd=ROOT,text=True).strip()
        claimrows.append(dict(claim=claim,source_file=source,source_commit=commit or 'Working tree reporting artifact; base abee084',dataset=dataset,metric_or_evidence=evidence,evaluation_protocol=protocol,evidence_strength=strength,limitations=limits,allowed_presentation_wording=allowed,wording_to_avoid=avoid))
    pd.DataFrame(claimrows).to_csv(WEEK/'CLAIM_EVIDENCE_MATRIX.csv',index=False)
    put(CLEAN,'DEPENDENCY_MAP.md','''# Dependency và artifact contracts

`IMPORT_EDGES.csv` là toàn bộ import edges tĩnh; `IMPORT_DEPENDENCY_CHECKS.json` ghi syntax, hardcoded-path locations và report-name collisions. AST không chứng minh dynamic imports resolve; CLI/import smoke và tests được ghi riêng. `NOTEBOOK_INVENTORY.json` kiểm tra cả9notebooks nhưng không execute để tránh ghi lại lịch sử.

| Nhóm | Code đọc | Inputs | Outputs lịch sử | Quy tắc |
|---|---|---|---|---|
| Original4branches | prepare/extract/evaluate/run_all; diagramo.branches/features | merged JSON, weights, baseline cache | results/cv, audit, figures | Không đổi công thức/split/encoder hoặc tái thêm experimental fusion |
| Kinematic XAI | kinematic_xai_common, diagramo.kinematic_xai | baseline OOF/checkpoints/schema | results/kinematic_xai | SHAP/raw-vs-probability đúng code, giữ task mapping |
| Temporal | raw_temporal*; temporal/temporal_v2 | original SVC,21channels, Train-only stats | raw_temporal*, locked_study | Segment attention khác task-level Transformer |
| AuditV3 | scripts/data_audit_v3 | metadata/source/cached arrays/OOF | results/data_audit_v3 | Diagnostic, không đổi clinical label |
| Provenance | scripts/provenance_verification | source docs + local metadata | results/provenance_verification | Explicit file-path imports; không import nhầm report.py |
| External | scripts/external_validation | pinned ZIP/XLSX/native reader | results/external_validation | Parser khác DiaGraMo physical preprocessing, không gộp semantics |
| Native | scripts/native_channel_research | Phase6 source/folds; private checkpoints/OOF | results/native_channel_research | Package-relative imports; no original14task head |
| Weekly cleanup | scripts/research_cleanup | read-only authoritative metrics/history | docs/research*, weekly_reports | Không gọi training runners |

`report.py` tồn tại ở nhiều thư mục. Baseline import bằng đường dẫn scripts; audit/provenance/external dùng explicit spec_from_file_location; native dùng relative package import. Giữ namespaces riêng; không deduplicate metrics superficially vì target, threshold, bootstrap và schema khác nhau. Hai requirements files có vai trò baseline lock và XAI additions, không coi là duplicate để xóa.
''')
    put(CLEAN,'CLEANUP_PLAN_VI.md','''# Cleanup đã thực hiện theo hướng bảo toàn

1. Chụp Git/branch/inventory/hashes trước source edits; private exact paths chỉ local.
2. KEEP_ACTIVE: source4branches, Temporal/audit/native modules, configs/CLIs và notebook entrypoints. KEEP_HISTORICAL: mọi report, OOF, checkpoint, fold assignment, snapshot và nghiên cứu trước.
3. REVIEW_FOR_REMOVAL: cache sinh ra, tmp, môi trường cũ hoặc duplicate byte-identical; chỉ lập đề xuất. Không xóa bất cứ tệp nào.
4. Giữ đường dẫn artifact; thêm research index/registry/timeline, báo cáo tuần11slides và figure/claim review.
5. Refactor tối thiểu CLI/test scratch với tests và behavior checks; không rewrite pipeline, gộp reader hoặc đổi model.
6. Hash kiểm tra sau cùng, so sánh metrics/OOF, notebook/config và privacy; không retrain/commit/push/merge tự động.
''')
    put(CLEAN,'REMOVAL_REVIEW_PLAN.md','''# Đề xuất review — chưa xóa

`DUPLICATE_ARTIFACT_REVIEW.csv` có paths an toàn/alias, size, hash, reason và dependency warning. Exact private paths ở ignored inventory để chủ sở hữu review local.

- Byte-identical historical reports/config snapshots/checkpoints: **KEEP_HISTORICAL mặc định**; bản sao có thể chứng minh provenance theo phase, không tự deduplicate.
- `__pycache__`, `.pytest_cache`, `tmp`: REVIEW_FOR_REMOVAL theo inventory; có thể tái tạo nhưng phải xác minh không còn job/lock phụ thuộc và xin owner approval trước.
- `.venv314`: chỉ review môi trường cũ sau khi xác minh kernel/notebook/automation không dùng; không xóa trong task này.
- Hai requirements files không duplicate semantics. `.cache` pretrained weights, embedding caches và raw recordings luôn giữ để tránh tải/trích xuất lại.
- Bốn tệp untracked có trước cleanup, gồm review note và completion watcher artifacts: giữ nguyên byte; không stage/xóa/di chuyển.

Điều kiện trước bất kỳ removal tương lai: explicit owner approval cho danh sách cụ thể, dependency/use check, và manifest/Git refs còn truy xuất được. Không có deletion approval trong task này.
''')
    print('Research docs, weekly report,11-slide content, figure index and12claims generated from existing evidence')


if __name__=='__main__':build()
