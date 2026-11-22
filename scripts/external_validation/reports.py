"""Vietnamese synthesis from persisted evidence; never change a locked experiment."""
import json
from pathlib import Path
import time
import pandas as pd


def compact_table(frame,columns=None):
    if columns is not None:frame=frame[columns]
    def value(x):
        if pd.isna(x):return 'NOT_COMPUTED'
        return f'{x:.6f}' if isinstance(x,float) else str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(map(value,row))+' |' for row in frame.itertuples(index=False,name=None)])


def scientific_analysis(c):
    start=time.monotonic();inv=c.load('DATASET_INVENTORY.json')
    c.table('complementarity_metrics.csv',[dict(status='NOT_COMPUTED',paired_participants=0,
        reason='No valid paired Kinematic/Temporal predictions; prior-only control is not either branch',
        both_correct=None,kinematic_only_correct=None,temporal_only_correct=None,both_wrong=None,
        corrected_by_fixed_fusion=None,harmed_by_fixed_fusion=None,probability_correlation=None)])
    domain=[
        dict(property='Cadence',diagramo='V3 median positive dt≈0.005 source-time-unit; rate≈200/source-time-unit',
            external=f"Candidate c2 median positive dt={inv['median_dt_candidate_native']} native ticks",comparison='NOT_PHYSICALLY_COMPARABLE',
            reason='Exact external clock unit unverified; no assumed /1000 conversion or Hz estimate'),
        dict(property='Coordinates',diagramo='Paper documents mm; model uses local original values',external='c0/c1 numeric ranges summarized; exact scale/axis unverified',
            comparison='UNVERIFIED',reason='Do not convert device units to mm without exporter/device-scale evidence'),
        dict(property='Pressure',diagramo='V3 summaries exist; physical calibration not independently certified',external='c6 candidate only; summary retained as native column',
            comparison='UNVERIFIED',reason='No force-unit scale or verified channel dictionary'),
        dict(property='Angles',diagramo='azimuth/tilt fields in source',external='c4/c5 candidate angle columns; convention unverified',comparison='UNVERIFIED',
            reason='No guessed swaps/deci-degree or altitude-to-tilt conversion'),
        dict(property='Negative time intervals',diagramo='V3 audit documented timestamp-safe masks; no new computation',external=f"{inv['negative_intervals']}negative candidate c2 deltas",comparison='DESCRIPTIVE ONLY',
            reason='Preserve acquisition order; do not sort or repair them'),
        dict(property='Missingness',diagramo='258complete14task participants',external=f"{inv['invalid_or_empty_recordings']}empty SVC; all120participants have a valid recording",comparison='DIFFERENT TASK CONTRACTS',
            reason='One raw file is empty, not an entire participant missing'),
        dict(property='Task duration/segment length',diagramo='Per-task recordings and computational segment boundaries known',external='Whole hw recording labels; no task boundary annotation; c2 unit unknown',comparison='NOT_COMPUTED',
            reason='Cannot compare whole-page/candidate recording duration with matched individual tasks'),
        dict(property='Pen up/down',diagramo='Explicit source pen_status',external='Candidate c3 binary in120valid SVC; transitions audited',comparison='SUPPORTED CANDIDATE ONLY',
            reason='Distribution does not independently verify encoding/schema'),
        dict(property='Relative coordinates',diagramo='Existing21channel original preprocessing',external='Native c0/c1 only; no task/channel semantic harmonization',comparison='NOT_COMPUTED',
            reason='No fabricated compatible representation')]
    c.table('domain_shift_summary.csv',domain)
    conf=[
        dict(dataset='DiaGraMo',participant_count=258,target_definition='Dataset-defined HD group incl.ongoing assessment',
            evidence='Protected V3 source tables',source_overlap='All outer Test cities already present in Train',
            class_support='City1:115group0/7group1; other14cities:0/136',negative_control_accuracy=.9728682170542635,
            inference='Strong source-label association; causal decomposition not identified',source_independent='NO'),
        dict(dataset='Drotar_Dobes_2020',participant_count=120,target_definition='Workbook DYSGR vs0; original study assessment',
            evidence='Original Methods and current source inventory',source_overlap='UNVERIFIED per-person sites not available',
            class_support='Paper describes case clinical-centre and control-school recruitment routes; actual site table unavailable',
            negative_control_accuracy=None,inference='Recruitment confounding risk; do not manufacture source labels from diag',source_independent='UNVERIFIED')]
    c.table('source_confounding_summary.csv',conf)
    limits=[
        ('RAW_SCHEMA','UNVERIFIED','Exact order/physical unit/export conventions missing; seven numeric columns alone insufficient','Author schema/exporter version and units'),
        ('TASK_MAPPING','INCOMPATIBLE for strict frozen transfer','14canonical vs8reported conditions; no per-recording task boundaries','Task-level annotations and genuinely shared stimuli/instructions'),
        ('EMPTY_RECORD','VERIFIED','One0sample SVC, alternate valid recording for sameID; no participant exclusion applied','Author explanation of repeated recording and official intended file'),
        ('CLOCK','VERIFIED numerical anomaly; units UNVERIFIED','Two candidate negative intervals; no sort/interpolation repairs','Clock reset/unit/session boundary documentation'),
        ('TARGET','NOT IDENTICAL BY DEFAULT','Slovak assessment vs DiaGraMo grouping incl.ongoing diagnosis','Harmonized independent clinical ascertainment'),
        ('SOURCE','UNVERIFIED independent-source validity','Per-person site/device/batch metadata absent; recruitment routes differ byclass','Both classes at several sources and source metadata'),
        ('PRIVACY','UNVERIFIED redistribution rights','Public download is not carte blanche to publish XML/free-text/linked OOF','Data owner permission/explicit data license'),
        ('PRIOR_METRICS','EXPLORATORY CONTROL ONLY','Class imbalance diagnostic; no sensor/motor information input','Admissible signal study after semantic gate'),
        ('UNCERTAINTY','CONDITIONAL','FixedOOF IIDbootstrap excludes training/selection/site/domain uncertainty','Independent sites and preregistered design')]
    c.table('limitations_matrix.csv',[dict(item=a,status=b,limitation=d,next_verification=e) for a,b,d,e in limits])
    scores=pd.read_csv(c.OUT/'model_metrics.csv')
    c.md('PHASE7_SCIENTIFIC_ANALYSIS.md',f'''# Phase 7 — Phân tích khoa học và negative controls

## Bằng chứng thực sự

Raw có thật, gồm120recording hợp lệ và một file0mẫu. Đã join đủ120ID/labels, giữ các recording của cùng người trong cùng candidate fold. Không có bảng engineered features phát hành trong3files nguồn đã tải. Không có task labels để tách8điều kiện viết được mô tả trong bài gốc.

Chỉ có prior-control OOF; không có kết quả Kinematic/Temporal Slovak. Do đó không thể kết luận Temporal tốt/kém hơn, không tính complementarity, corrected/harmed hoặc fusion. `complementarity_metrics.csv` ghi NOT_COMPUTED và paired count0, không điền số0 giả cho các đại lượng chưa tính. Không so sánh prior-control Slovak với0.8472F1 của DiaGraMo như paired test.

{compact_table(scores[scores.metric.isin(['accuracy','f1_macro','auc','balanced_accuracy','sensitivity','specificity','brier','log_loss','ece10'])],['model','participant_count','completed_folds','metric','value','ci95_low','ci95_high','fold_mean','fold_sd','evaluation_scope'])}

Pooled prior AUC≈0.48 trong khi mỗifold AUC0.5: Trainprior thay đổi đối ứng với tỷ lệ nhãn heldoutfold. Đây là hiệu ứng thiết kế của control, không phải học tín hiệu âm. ECE10=0 do các xác suất cùngbin và meanprior khớp tỷ lệ nhãn tổng; điều đó không chứng minh một mô hình lâm sàng được calibration tốt. Sensitivity0 và specificity1 phản ánh luôn dự đoán nhóm0.

CI95% là percentile2000lần resample các participant OOF với xác suất cố định. Không resample modeltraining; chưa xử lý siteclustering vì thiếu source metadata. Một số CI control suy biến vì cùng dự đoán một lớp; không xem là độ chắc chắn về hiệu năng trên nguồn mới.

## Domain shift và source

`domain_shift_summary.csv` tách quan sát native/candidate với đại lượng vật lý. Median c2delta8 không tự có nghĩa8ms/125Hz; so trực tiếp với0.005source-time của DiaGraMo là không hợp lệ. Không quy mọi khác biệt pressure/góc/duration cho hardware. Không gán nghĩa motor sinh học cho cột chưa được xác nhận.

Original Methods mô tả tuyển case ở cơ sở special-needs, controls ở trường; dataset thứ hai vì vậy không tự giải quyết source confounding. Source×class table ở từngngười không có; không tạo source bằng cách sao label rồi báo perfectnegativecontrol. `source_confounding_summary.csv` phân biệt V3observations với documentary risk ởSlovak.

## Quyết định

**NO-GO hiện tại cho signal-model transfer/comparison**; **CONDITIONAL GO** sau khi tác giả xác nhận exactexport schema/units và task/sessionmapping. Không có thất bại performance Temporal để giải thích: nó chưa chạy, nên không đổ cho inadequatelearning/nhỏN. Đây là thiếu điều kiện thực nghiệm, không phải một kết quả mô hình âm.
''')
    c.status(7,'COMPLETE',start,dict(signal_model_comparison='NOT_COMPUTED',prior_control='5fold completed',
        complementarity='NOT_COMPUTED',domain_shift='Native descriptive only',source='Unverified source independence'),
        dict(no_cross_cohort_paired_tests=True,no_causal_claims=True),
        ['PHASE7_SCIENTIFIC_ANALYSIS.md','complementarity_metrics.csv','domain_shift_summary.csv','source_confounding_summary.csv','limitations_matrix.csv'])


def final_report(c,budget):
    start=time.monotonic();inv=c.load('DATASET_INVENTORY.json');ev=c.load('experiment_validation.json')
    metrics=pd.read_csv(c.OUT/'model_metrics.csv').set_index('metric')
    xml=c.load('XML_SCHEMA_SUMMARY.json');lock=c.load('PREEXPERIMENT_DECISIONS.json')
    source_rows=[
        dict(source='Original paper',url='https://doi.org/10.1038/s41598-020-78611-9',location='Methods; Table1/2; supplementary information; Data availability',
            evidence='Recruitment,120children, device family,8tasks, clinical assessment; not an export unit dictionary',status='VERIFIED document'),
        dict(source='Original author repository',url=f'https://github.com/{c.REPO}/tree/{c.COMMIT}',location='Three-file listing; README; pinned Git blobs',
            evidence='Exact downloaded sources; no LICENSE/schema files in pinned tree',status='VERIFIED'),
        dict(source='Supplement1',url='https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-020-78611-9/MediaObjects/41598_2020_78611_MOESM1_ESM.pdf',
            location='Single-page writing template; locally rendered and visually inspected',evidence='l/le atnormal/fast, words/pseudoword/sentence; no rawtaskcutpoints',status='VERIFIED document'),
        dict(source='Original repo schema issue',url=f'https://github.com/{c.REPO}/issues/2',location='Issue2; comments count0 on check',
            evidence='A reader asks meanings of7headerless columns; no author schema answer found. Not itself proof of channel meaning.',status='VERIFIED access gap'),
        dict(source='DiaGraMo data descriptor',url='https://doi.org/10.1038/s41597-026-08625-x',location='Protected Phase5source review; Methods/Table3',
            evidence='Different clinical grouping/task/device contract',status='VERIFIED previous review')]
    for row in source_rows:row['checked_utc']=c.stamp()
    c.table('SOURCE_DOCUMENTS.csv',source_rows)
    paper=c.LOCAL/'paper.html';supplement=c.LOCAL/'supplement.pdf'
    c.write('DOCUMENT_CHECKSUMS.json',dict(files=[dict(alias=p.name,sha256=c.sha(p),bytes=p.stat().st_size) for p in [paper,supplement] if p.exists()],
        note='Documents cached local-only; dataset files separately hashed'))
    blocker='''# Blockers

1. **Exact SVC export schema/units UNVERIFIED:**7numericcolumns have no header. c2clock/c3pen/c4-c5angles/c6pressure are supported candidates, not confirmed exporter conventions. Do not use guessed /1000 time scaling or altitude/tilt swaps. Originalrepo issue2 has no author answer.
2. **Task contract incompatible:**8conditions in paper vs14canonical DiaGraMo tasks. Actual SVCs have hw00001/one extra hw00002, no task boundaries. No exactsharedtask mapping; do not split equal-length blocks or infer tasks frompageposition.
3. **One zero-sample SVC and two negative candidate time deltas:** all120participants retain another valid recording; no source repair/exclusion applied. Need intendedsession/recording policy before signaltraining.
4. **Source/target transportability:** per-person site/device/batch absent. Case clinical-centre vs controlschool recruitment documented; no sourceheldout validation. Slovakdiag not automatically identical to DiaGraMogroup including ongoingassessment.
5. **Release restrictions:** no explicit standalone datasetLICENSE found. Boundedlocaldownload was explicitly requested and publicpaper-linked; no redistribution of raw, clinical metadata or individual OOF. XMLcontains history/comments; do not publish them.

These blockers stop signal-model operations, not structural audit, label joins, priorcontrol, tests or reporting. OutcomeE applies to learning/transfer under current evidence; rawtrajectories do exist. No fabricated features, time units, task indices, zerochannels, models or metrics.
'''
    c.md('BLOCKERS.md',blocker)
    c.md('NEXT_RESEARCH_DECISION_VI.md','''# Quyết định nghiên cứu tiếp theo

**Ưu tiên duy nhất:** nhờ owner xin tác giả một gói schema/exporter không chứa dữ liệu cá nhân: thứ tự và đơn vị7cộtSVC, timestamporigin/reset, angularconventions, ý nghĩa hw00001/hw00002, taskcutpoints và file0mẫu có chủ đích hay không. Kèm hỏi quyền dùng/derivedrelease và source metadata. Không gửi email tự động.

**NO-GO hiện tại** cho exactfrozentransfer, sharedtasktransfer và Kinematic–Temporal signalcomparison. Córaw không đồng nghĩa khớp14taskcontract. Không nên đầu tư8giờGPU để tối ưu trên schemachưaverified.

**CONDITIONAL GO choD** sau khi xác nhận schema: khóa một benchmarkparticipantStratified5/seed42 riêng trong cohortSlovak, compacthandcrafted và compacttemporal, cùngchannel/taskcontract và matchedbudget. Không gọi đây là zero-shotDiaGraMo. Nếu không có taskcutpoints nhưng schemađủ, protocoltask-agnosticwhole-recording cần được viết riêng trướcscores và mô tả khác mô hình14task. Không đổi recipe hiện tại dựa trên priorcontrol.

**A/B** vẫn cần thật sự cùng tác vụ và fullsemanticcontract; labels/source cân bằng và external lockbox để kiểm tra scientificgeneralization. AdditionalCNN/Transformer/SSL không thể tự khôi phục tasklabels/clockprovenance bị thiếu. Nếu tác giả không cung cấp schema, đóng nghiên cứu hiện tại như compatibility/no-go report, giữ priorcontrol chỉ để minh họa classbalance, không làm headlineclinicalmetric.
''')
    c.md('REPRODUCE.md','''# Reproduce — bounded Phase6 investigation

Chạy từ projectroot bằng virtualenv hiện tại; không chạy prepare.py, evaluate.py, báo cáo cũ hoặc fullPhase4queue.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/external_validation/run.py --phase all --max-runtime-hours 8 --report-reserve-minutes 15
```

Các phase CLI:0,1,continue,validate. All/continue kiểm tra lockedrecipe và resumes priorfolds đã hoàn tất, không refit. SESSION_CLOCK giữ mốc toàn session qua các CLI invocation; không reset budget bằng việc tách lệnh. Muốn một session mới sau khi hết giờ cần protocol vận hành riêng, không tự xóa clock để kéo dài budget. Budgetreserve phục vụ tests/reporting, không fillGPUtime.

Nguồnpincommit0079fdd34fc5eff0c1c58f9689127b2c73b57f8b, files bounded≤32MiB, Gitblob/SHA256/sizechecks, ZIPpath/CRC/size/symlink/executableguard. Filesđãcó phải khớp hash; extractexistingfiles khác bị từ chối. Binaryfetch dùng publicrawGitHub vì connector chỉUTF8. Download permissions chỉ cho localread theo task; không redistribution.

Excel được đọc OOXML bằng stdlib: giữrawcachedvalues, không executeformulas. Một directlabelreference được đối chiếu cachedvalue. IDnumeric và text được canonicalizezero-padding label-independent; collision/missinglabels bị từ chối. Không dùng XMLcommenthistory làm feature/label; chỉ locally kiểm tra nguồn.

ReaderSVC độc lập, trả nativec0…c6, rejects0samples/nonfinite/header mismatch; không gọi baselinepreprocessing. No time/angleconversion/windowing/scaler/rawtraining. Candidateclock/pen diagnostics không phải verifiedphysicalschema.

LOCKED_CONFIG được tạo trướcscores; priorDummyfit Train-only,5participantfolds,seed42,thresholdp>0.5. JSONclasscountcheckpoint và SHA/split/configsignatures; no optimizer/RNG N/Aclosed-form. IndividualOOF/splits/metadata/checkpoints trong data/external_datasets đãGitignore. Publictables chỉaggregatefold/cohortresults. Không fusion hoặc Kinematic/Temporalfit.

PaperHTML/Supplement được kiểm tra bằng publicprimaryURLs trong SOURCE_DOCUMENTS.csv; caches chỉlocal. Supplement1trang render và xem trực tiếp; không suy taskboundaries từ ảnh. Reproduce từclonekhác cần tài liệu cached hoặc checkURLlại; no automatic access to request-onlyFrenchcohorts.

Tests/validation/manifests tạo dưới results/external_validation. Finalintegrityhash kiểm tra toàn protectedmanifest, checkpointdata + sourcefiles. Không sửa baseline/notebooks/oldreports, không commit/push/merge. Các mụcNOT_RUN không được đổi thànhPASS.
''')
    h=c.load('HARDWARE_BUDGET.json')
    scores=metrics.loc[['accuracy','f1_macro','auc','balanced_accuracy','sensitivity','specificity','brier','log_loss','ece10']].reset_index()
    scope='Phạm vi: prior-control, 120 participants, target Excel DYSGR vs 0, participant Stratified 5-Fold/seed 42, đủ 5/5 folds. Đây là exploratory internal CV của external dataset, không phải external validation của mô hình DiaGraMo.'
    sections=[
        ('Executive Summary',f"""**Có raw thật, nhưng chưa có điều kiện hợp lệ để so sánh Kinematic–Temporal hoặc đánh giá frozen DiaGraMo trên dữ liệu này.** Đã thực hiện tải nguồn, audit toàn bộ raw, đối chiếu nhãn, kiểm tra compatibility, reader độc lập, khóa protocol, chạy đối chứng, phân tích và kiểm thử. Không đánh dấu câu hỏi motor representation là đã được giải quyết.

Có {inv['svc_recordings']} SVC: {inv['valid_recordings']} recording hợp lệ và {inv['invalid_or_empty_recordings']} file không có mẫu. Cả 120 ID vẫn có một recording hợp lệ; tổng {inv['samples']:,} samples. Excel có 57 DYSGR và 63 nhãn 0. Bài báo mô tả tám điều kiện viết, nhưng raw không có annotation để tách các điều kiện đó. Chỉ Train-prior label control được chạy; không dùng GPU huấn luyện."""),
        ('Independent Dataset Provenance',f"""Nguồn chính là [Drotár & Dobeš (2020)](https://doi.org/10.1038/s41598-020-78611-9), Methods, Table 1/2, Supplement và [repository tác giả](https://github.com/{c.REPO}/tree/{c.COMMIT}). Commit nguồn `{c.COMMIT}` được khóa trước tải dữ liệu. Không thay thông tin tuyển mẫu bằng mô tả của paper khác.

Methods ghi nhận ba chuyên gia đánh giá và sử dụng Wacom Intuos Pro Large. Case được tuyển qua cơ sở special-needs, đối chứng qua trường học; đây là nguy cơ confounding tiếp tục cần xem xét. Supplement một trang đã được tải, render và xem trực tiếp. Hai ứng viên Pháp chỉ dùng làm bối cảnh đã kiểm tra ở Phase 5, không tải dữ liệu bị hạn chế access hoặc giả định chúng độc lập với nhau."""),
        ('Actual Files and Data Schema',"""Ba file nguồn đã được xác minh kích thước, Git blob và SHA-256: README, Excel 17.201 bytes, ZIP 16.731.844 bytes. ZIP có 365 members/124 files: 121 SVC, hai XML và một .DS_Store; tổng 81.422.448 bytes giải nén theo inventory. Đã kiểm tra traversal, absolute paths, symlink, CRC, expansion limits và executable trước extraction. Không thực thi code hay deserialize pickle.

SVC có header đếm mẫu, sau đó là các hàng bảy giá trị số, không có tên cột. Excel có một sheet và 1.621 stored cells nhưng chỉ 120 ID có dữ liệu; các ô trống hoặc chỉ formatting không tạo thêm participant. Xem external_file_inventory.csv và external_schema_inventory.csv; đường dẫn từng recording và nội dung cá nhân chỉ giữ local."""),
        ('Participant and Label Inventory',"""Đã join đủ 120 ID giữa Excel, filename và parent directory. Excel trộn ID dạng text năm chữ số với 49 ID dạng số ba chữ số. Quy tắc zero-padding thành năm chữ số áp dụng độc lập nhãn, không collision và khớp đủ SVC; ID nguyên gốc giữ local để đối chiếu.

Target là diag: DYSGR=1, 0=0. Một ô nhãn có công thức tham chiếu ô cùng cột: cached value đã được đối chiếu, không thực thi công thức. XML hiện tại có 192 ID và khớp 120 ID nghiên cứu; XML cũ có 155 ID, khớp 89. Không dùng các ID thử nghiệm/ngoài cohort làm subject mới. Nhãn Slovak không tự tương đương với group của DiaGraMo, vốn bao gồm ongoing assessment."""),
        ('Raw Signals and Acquisition Quality',f"""Đã khảo sát toàn bộ SVC. {inv['valid_recordings']} recording có dữ liệu qua kiểm tra header/column count/finite values, gồm {inv['samples']:,} samples. File 0 mẫu bị reader từ chối; ID tương ứng có một recording hợp lệ khác, nên không loại participant. Không ghi đè hoặc sửa file nguồn.

Cột c2 là timestamp candidate: median positive Δt=8 native units, có hai Δt âm và không có Δt=0. Có 2.430 khoảng lớn hơn mười lần median riêng recording; đây là diagnostic, không phải quy tắc sửa dữ liệu. Cột c3 binary trong các recording hợp lệ. Mọi nhãn semantic của cột và đơn vị vật lý vẫn cần xác nhận. Không sort, resample hoặc nội suy; channel summaries được tổng hợp theo participant."""),
        ('Source–Label Confounding',"""Tài liệu mô tả tuyển case tại cơ sở special-needs và đối chứng tại trường, nên dataset thứ hai không tự giải quyết source confounding. Không có city/site/device serial/batch/date theo participant để tạo source-held-out split hoặc tính source-only negative control hợp lệ.

Không tạo source bằng cách sao chép nhãn bệnh, không suy địa điểm từ ID, không báo một perfect source control giả. Source independence và đóng góp nhân quả của nguồn vẫn UNVERIFIED. Age/sex/hand và XML comments không được dùng làm predictors."""),
        ('DiaGraMo vs External Dataset Compatibility',"""Giữ nguyên Kinematic 14×74=1.036 và Stroke với bảy kênh gốc, 21 kênh xử lý, window tối đa 128 mẫu, kernels 5/13/29, bốn token 32 chiều, attention hai heads, task identity 14 và learned participant pooling. Không sửa kiến trúc hoặc normalization gốc.

External contract chưa có 14 task identities/instructions hay channel semantics/units được xác minh. A: NOT VALID; B: NOT FEASIBLE; C: BLOCKED; D chỉ cho phép prior label control, còn mô hình tín hiệu BLOCKED. Xem TRANSFER_PROTOCOL_DECISION.json. Dataset classification A (có raw) khác với research Outcome E (chưa có đường học tín hiệu được chấp nhận)."""),
        ('Task Mapping and Semantic Limitations',"""Đã xét đủ 112 cặp tác vụ theo nội dung, motor complexity, speed, pen lifts, duration, instruction, grade và language. Không có EXACT MATCH. Copy–sentence chỉ là CLOSE MOTOR ANALOGUE; loops–l/le chỉ WEAK ANALOGUE.

Filename hw không cung cấp annotation tám điều kiện viết. Không gán task indices bằng tên tương tự, thứ tự trên trang hoặc cắt thành tám đoạn bằng nhau. Nếu sau này có annotation, task-subset model là mô hình/protocol mới và cần development training riêng; không gọi nó là frozen model 14 tác vụ."""),
        ('Channel Harmonization',"""external_reader.py trả đúng native c0…c6, bảo toàn thứ tự và terminal sample của recording hợp lệ; đọc lại cho kết quả giống hệt. API không nhận nhãn. File 0 mẫu bị từ chối rõ ràng.

Không có physical conversion, missing-channel imputation, fitted scaler hoặc raw-model windows. Pattern số hỗ trợ một vài candidate meanings nhưng không xác minh clock unit, coordinate scale, thứ tự góc hay altitude-to-tilt. Issue 2 trong repository gốc hỏi về bảy cột không header và chưa có câu trả lời tác giả. Generic SVC reader hoặc code thứ cấp không thay bằng chứng của exporter cụ thể. Window/mask tests ghi NOT_RUN thay vì PASS."""),
        ('Locked Research Protocol',f"""Recipe SHA-256 `{lock['locked_config_sha256']}` được lưu trước khi tính score. Dùng participant Stratified 5-Fold, shuffle=True, seed 42; mọi recording cùng ID nằm cùng fold. Prior control dùng DummyClassifier(strategy=prior), constant X, chỉ fit tỷ lệ nhãn Train và ngưỡng cố định p>0.5.

Không hyperparameter search, scaler, early stopping, calibration fit hoặc Test-based selection. Không GAF/CWT/SSL/fusion. Khoá 2.000 bootstrap resamples của các hàng OOF theo participant; không dùng kết quả để đổi recipe. Các bước học Kinematic/Temporal không được cấp phép thực thi bởi gate semantic hiện tại."""),
        ('Executed Experiments',f"""Chỉ `Train_prior_label_control` đã fit đủ 5/5 folds. JSON checkpoint lưu class counts, private split IDs và config hash; SHA và exact replay đã kiểm tra. Không refit fold hoàn thành khi chạy resume. Với closed-form estimator, optimizer/RNG state không áp dụng; không giả báo đã kiểm tra neural resume.

{scope}

{compact_table(scores,['model','participant_count','completed_folds','metric','value','ci95_low','ci95_high','fold_mean','fold_sd'])}

Individual OOF, split IDs và checkpoint chỉ lưu trong data đã Git-ignore. Các bảng public là aggregate. Mọi mô hình tín hiệu chưa chạy đều có lý do trong experiment_registry.csv; không dùng score 0 thay cho NOT_COMPUTED."""),
        ('Kinematic vs Temporal Results',"""**NOT_COMPUTED.** Chưa có hai mô hình tín hiệu được đánh giá hợp lệ trên Slovak. Không so prior control Slovak với Kinematic/Temporal DiaGraMo như paired observations hoặc cùng target. Không có bằng chứng mô hình nào tốt hơn, và không suy failure mode của Temporal từ việc chưa chạy nó."""),
        ('Cross-Dataset Transfer Results, if valid',"""Không có cross-dataset transfer hợp lệ được chạy. Không load hoặc rerun frozen checkpoints, không fit scaler mới trên external Test, không fine-tune hay điều chỉnh threshold bằng nhãn ngoài. Prior control nội bộ Slovak không phải external validation của trained DiaGraMo."""),
        ('Error and Complementarity Analysis',"""Không có paired pK/pT nên both-correct, Kinematic-only, Temporal-only, both-wrong, corrected/harmed do fusion và probability correlation đều NOT_COMPUTED. Bảng complementarity ghi paired_participants=0 và các đại lượng chưa tính để trống, không điền số 0 giả.

Prior control có TN=63, FP=0, FN=57, TP=0 vì luôn dự đoán nhóm 0. Đây là tác động class balance của đối chứng không có tín hiệu, không phải attribution lỗi vận động. Không bootstrap chênh lệch giữa hai cohort khác nhau."""),
        ('Statistical Uncertainty',"""CI95% là percentile từ 2.000 lần resample participant OOF với xác suất cố định. Nó không bao hàm training variance, researcher selection, source clustering hoặc domain shift. Thiếu site metadata nên không thể bootstrap source clusters; giả định IID có thể làm bất định nhỏ hơn thực tế.

Pooled prior AUC khoảng 0,47995 trong khi từng fold AUC=0,5: tỷ lệ nhãn Train thay nhẹ theo chiều ngược tỷ lệ held-out. ECE10=0 do các xác suất nằm trong cùng bin và trung bình khớp class balance, không chứng minh clinical calibration tốt. CI sensitivity/specificity suy biến là đặc tính always-predict-zero, không phải độ chắc chắn khi chuyển sang nguồn mới."""),
        ('Domain Shift and Source Robustness',"""Không diễn giải 8 native ticks thành 8 ms hoặc 125 Hz. Không so trực tiếp các native pressure/angles/duration với DiaGraMo sau preprocessing. domain_shift_summary.csv phân biệt quan sát số, giả thuyết semantic và thông tin vật lý chưa xác minh.

Không quy khác biệt cho hardware khi chưa có cross-device hoặc session metadata. Cả hai nghiên cứu đều có nguy cơ recruitment confounding. Chưa có bằng chứng về unseen-source robustness hoặc motor biomarkers độc lập nguồn."""),
        ('Compute and Reproducibility',f"""Máy có {h['cpu_physical']} nhân vật lý/{h['cpu_logical']} luồng, RAM tổng {h['ram_total_bytes']/2**30:.2f} GiB, Quadro T1000 4 GB và CUDA available. Trần session tám giờ, reserve báo cáo 15 phút, tối đa một GPU worker. Thực tế không khởi chạy GPU training worker nào.

Prior fits cộng metrics/bootstrap tốn {ev['wall_seconds']:.3f} giây wall và {ev['cpu_seconds']:.3f} giây process CPU. RSS lớn nhất quan sát tại các checkpoint đo là {ev['peak_observed_process_rss_bytes']/2**20:.1f} MiB; không gọi đây là OS-wide peak. GPU training và tensor allocation của thí nghiệm bằng 0. Resource query/CUDA context không phải training. Phase status JSON giữ thời gian từng bước; tổng elapsed session bao gồm đọc nguồn, viết và kiểm tra tài liệu, không nhầm với thời gian fit."""),
        ('Privacy and License Restrictions',"""Download local có giới hạn được yêu cầu rõ trong task và nguồn nằm trong public repository được paper liên kết. Không dùng paid API/cloud GPU. Không tìm thấy standalone data LICENSE trong tree ba file; không tự áp giấy phép bài báo cho mọi nội dung XML hoặc derived clinical outputs.

Raw, participant clinical metadata, exact recording paths, splits và individual predictions đều giữ local Git-ignored. XML history/comments cần được bảo vệ. Không copy chúng sang báo cáo, không gửi email, commit/push/merge, đổi visibility hoặc rewrite history. Xem inventory và kiểm tra staging cuối cùng."""),
        ('Scientific Validity and Limitations',"""Hoàn tất phần audit có thể thực hiện không đồng nghĩa hoàn tất so sánh representation. Grouping theo project ID được kiểm chứng; independent physical identity và source exchangeability chưa được xác nhận độc lập. Metadata/source coverage không đủ cho source-held-out claims.

Tám điều kiện trong paper, task boundaries thiếu, schema/units chưa rõ, một recording rỗng và hai candidate negative intervals đều được ghi nhận. Không có poor Temporal score để đổ cho thiếu capacity hoặc sample size: Temporal chưa chạy. Prior control không trả lời câu hỏi temporal information beyond kinematics."""),
        ('Recommended Next Research Direction',"""Ưu tiên xin tác giả schema/exporter cho bảy cột và đơn vị, clock resets, angle conventions, ý nghĩa hw codes, task boundaries, official recording-selection policy và quyền dùng/derived release. Không gửi tự động.

Sau clarification, có thể khóa protocol D nhẹ trên Slovak hoặc một protocol task-agnostic riêng trước score, với kinematic và temporal matched inputs/budget. Chỉ mở exact frozen transfer khi full semantic contract thực sự tương thích. **CONDITIONAL GO** cho clarification và nghiên cứu phương pháp; **NO-GO hiện tại** cho signal-model comparison, clinical validation hoặc source-invariant claims. Không chạy thêm GPU chỉ vì budget còn."""),
    ]
    c.md('EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md','# Phase 6 — Independent Dataset Compatibility & Controlled Research\n\n'+
        '\n\n'.join(f'## {i}. {title}\n\n{text}' for i,(title,text) in enumerate(sections,1)))
    c.md('EXECUTIVE_SUMMARY_VI.md',f"""# Tóm tắt Phase 6

**Có raw Slovak thật, nhưng chưa đủ điều kiện đánh giá Kinematic/Stroke Transformer theo contract DiaGraMo.** Đã thực hiện tải nguồn, audit, đối chiếu nhãn, compatibility, adapter số, khóa protocol, đối chứng và kiểm thử. Chưa push/merge.

ZIP có121SVC cho120ID:120recording hợp lệ và một file0mẫu; mọi người vẫn có một recording hợp lệ. Tổng {inv['samples']:,} samples trong bảy cột số. Excel có57DYSGR/63zero. HaiXML chứa192/155ID, không phải cohortcounts. Paper mô tả tám điều kiện viết nhưng không cung cấp task boundaries trong raw. Hai candidate time deltas âm được giữ nguyên. Thứ tự cột và physical units chưa được tác giả xác nhận.

**Đã chạy:** Train-prior label control,5/5participant folds,seed42, chỉ dùng class counts Train. Accuracy0,525; Macro-F1 {metrics.loc['f1_macro','value']:.6f}, CI95% [{metrics.loc['f1_macro','ci95_low']:.6f}; {metrics.loc['f1_macro','ci95_high']:.6f}]; balanced accuracy0,5, sensitivity0, specificity1. Đây là đối chứng class balance, không phải Kinematic hoặc Temporal. Pooled AUC≈0,48 do thay đổi prior giữa các folds, không phải motor information.

**Chưa chạy:** frozen transfer, shared-task transfer, raw Kinematic/compact Temporal và fusion/complementarity. Chúng bị chặn bởi semantic gate, không phải hết GPU budget. Prior fits cộng metrics/bootstrap dùng {ev['wall_seconds']:.3f} giây wall/{ev['cpu_seconds']:.3f} giây CPU; RSS lớn nhất quan sát {ev['peak_observed_process_rss_bytes']/2**20:.1f}MiB. GPU training bằng0.

Nguy cơ confounding vẫn còn: case/control đi qua các nguồn tuyển khác nhau; thiếu per-person source metadata để kiểm chứng held-out source. Public download không chứng minh quyền redistributing mọi raw/clinical/OOF artifact. Các hồ sơ cá nhân chỉ giữ trong vùng local đã Git-ignore.

**Quyết định:** dữ liệu thuộc classificationA (có raw), nhưng research OutcomeE cho signal-model execution với bằng chứng hiện tại. Phần điều tra có thể thực hiện đã hoàn tất; câu hỏi motor representation chưa giải quyết. Ưu tiên duy nhất: xin tác giả exact column schema/units và task/session mapping trước khi khóa một benchmark mới. Tests và hash checks thực xem VALIDATION_CHECKS.json; không coi NOT_RUN là PASS.
""")
    c.status(8,'COMPLETE',start,dict(report_sections=20,full_signal_research_complete=False),
        dict(report_from_persisted_metrics=True,no_fabricated_signal_metrics=True),
        ['EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md','EXECUTIVE_SUMMARY_VI.md','NEXT_RESEARCH_DECISION_VI.md','BLOCKERS.md','REPRODUCE.md','SOURCE_DOCUMENTS.csv','DOCUMENT_CHECKSUMS.json'])
