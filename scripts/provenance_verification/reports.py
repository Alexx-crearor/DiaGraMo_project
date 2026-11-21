"""Evidence synthesis, separate from read-only checks and protected historical reports."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def build(v):
    out=v.OUT
    local=json.loads((out/'LOCAL_SOURCE_CHECKS.json').read_text(encoding='utf-8'))
    rds=json.loads((out/'RDS_VERIFICATION_STATUS.json').read_text(encoding='utf-8'))
    now=datetime.now(timezone.utc).isoformat()
    paper='https://doi.org/10.1038/s41597-026-08625-x'
    zen='https://zenodo.org/records/21236910'
    refs=f'Nguồn chính: [data descriptor]({paper}), [Zenodo v2]({zen}). Kiểm tra {now}. Attribution: Zvončáková, Mekyska, Klocek, Mucha & Galáž (2026); dataset CC BY 4.0. Bảng kết quả local trỏ về Data Audit V3 tại commit f419530; không tái chạy V3.\n'

    def md(name,text):
        (out/name).write_text(text.strip()+'\n',encoding='utf-8')

    def grid(rows):
        if not rows:return 'Không có hàng.'
        keys=list(rows[0])
        return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(str(r.get(k,'')).replace('|','/').replace('\n',' ') for k in keys)+' |' for r in rows])

    documents=[
        dict(source_id='S01',url_or_doi=paper,title='Multimodal Czech online handwriting and cognitive data from children with and without handwriting disabilities',authors='Zvončáková; Mekyska; Klocek; Mucha; Galáž',year=2026,supports='Recruitment, clinical assignment, privacy, protocol, omission choices',evidence_location='PDF pp.2–6,8–10; Assignment of diagnosis; Ethics; Usage Notes',access='PDF fetched successfully by requests; HTML web redirect blocked',cache='paper.response'),
        dict(source_id='S02',url_or_doi=zen,title='DiaGraMo dataset v2',authors='Zvončáková et al.',year=2026,supports='Version2 privacy changes; file checksum; publication DOI',evidence_location='Notes; metadata.license; files[0]; conceptrecid',access='Record HTML and API inspected; dataset ZIP NOT downloaded',cache='zenodo.response'),
        dict(source_id='S03',url_or_doi='https://zenodo.org/api/records/18299326',title='Concept DOI resolves to current version',authors='Zenodo / dataset authors',year=2026,supports='Concept18299326 maps to version21236910; not two unrelated datasets',evidence_location='id; conceptrecid; links',access='API inspected',cache='concept.response'),
        dict(source_id='S04',url_or_doi='https://github.com/BDALab/HandAQUS_for_Cintiq',title='HandAQUS for Cintiq',authors='BDALab',year='not inferred',supports='Wintab-based Cintiq acquisition software; not per-session device identity',evidence_location='README; license',access='Official README read; no installer downloaded',cache='handaqus.response'),
        dict(source_id='S05',url_or_doi='https://github.com/BDALab/handwriting-sample',title='handwriting-sample',authors='BDALab',year='not inferred',supports='Seven channels, validation and unit conversion support',evidence_location='README Input data; SVC example; unit transformation',access='Official README read; current version not proof of original pinned version',cache='sample.response'),
        dict(source_id='S06',url_or_doi='https://doi.org/10.1038/s41598-020-78611-9',title='Dysgraphia detection through machine learning',authors='Drotár; Dobeš',year=2020,supports='Independent Slovak online cohort; clinical/source limitations',evidence_location='Methods; Table1; author repository link',access='Publisher HTML and PDF read',cache='slovak.response'),
        dict(source_id='S07',url_or_doi='https://github.com/peet292929/Dysgraphia-detection-through-machine-learning',title='Original Slovak dataset repository',authors='peet292929 / Drotár and Dobeš',year=2020,supports='Public data links exist; explicit data license not confirmed',evidence_location='README; repository file listing',access='Read-only web inspection; no data downloaded',cache=''),
        dict(source_id='S08',url_or_doi='https://doi.org/10.1038/s41746-018-0049-x',title='Automated human-level diagnosis of dysgraphia using a consumer tablet',authors='Asselborn et al.',year=2018,supports='French online BHK cohort and restrictions',evidence_location='Participants; DATA AVAILABILITY p.9',access='PDF fetched; PyMuPDF fallback after pypdf token-limit error',cache='asselborn.response'),
        dict(source_id='S09',url_or_doi='https://www.mdpi.com/1424-8220/21/21/7026',title='Analysis of Graphomotor Tests with Machine Learning Algorithms for an Early and Universal Pre-Diagnosis of Dysgraphia',authors='Devillaine et al.',year=2021,supports='Graphomotor cohort; device/source differences; request-only access',evidence_location='Sections2.1,2.2,2.4; Data Availability Statement',access='Web HTML read successfully; direct requests got403',cache=''),
        dict(source_id='S10',url_or_doi='https://data.mendeley.com/datasets/39hr8dx76p/1',title='Potential Dysgraphia Handwriting Dataset of School-Age Children',authors='Ramlan',year=2023,supports='Offline labels; CC BY4.0; not raw temporal compatible',evidence_location='Description; Licence',access='Official landing page inspected; no data downloaded',cache=''),
        dict(source_id='S11',url_or_doi='https://doi.org/10.1016/j.dib.2024.110534',title='Development of potential dysgraphia handwriting dataset',authors='Ramlan et al.',year=2024,supports='83participants/249images; distinguish image counts and participants',evidence_location='Abstract; Specifications Table',access='Primary paper via PMC inspected',cache=''),
        dict(source_id='S12',url_or_doi='local:scripts/prepare.py',title='Protected cohort implementation',authors='Repository implementation',year='local',supports='14 task canonicalization; group eligibility; no omit exclusion',evidence_location='TASKS; canonical; eligible expression; meta selection',access='Read only',cache=''),
        dict(source_id='S13',url_or_doi='local:results/data_audit_v3/',title='Protected Data Audit V3',authors='Repository audit',year=2026,supports='Observed source association; fidelity; current cohort and OOF',evidence_location='COMPREHENSIVE_DATA_AUDIT_VI.md; PhaseC/D; source tables',access='Read only; no rerun',cache=''),
        dict(source_id='S14',url_or_doi='https://zenodo.org/records/21236910/preview/DiaGraMo-project.zip?include_deleted=0',title='Dataset archive documentation preview',authors='Zenodo',year=2026,supports='Access blocker: Too many files inside ZIP; codebook and R sources not recovered',evidence_location='Preview error; container API403',access='Preview fails; no archive or participant records fetched',cache='preview.response')]
    for row in documents:
        path=v.PRIVATE/row.pop('cache')
        row['checked_utc']=now
        row['cached_sha256']=v.sha(path) if path.is_file() else ''
        row['verification_status']='UNVERIFIED' if row['source_id']=='S14' else 'VERIFIED'
    v.table('SOURCE_DOCUMENTS.csv',documents)

    acquisition=[
        ('Population/recruitment','Schools and pedagogical-psychological counselling centres; different routes for HD and TD','S01 pp.2–3','VERIFIED','No city-by-city sampling frame or referral denominator'),
        ('City identity','Pseudonymized city labels in released metadata','S01 Ethics; S13','VERIFIED','Real cities and device mappings unavailable; city is not hospital/device'),
        ('Clinical source','Prior PPCC records and school anamnestic reports; includes ongoing diagnoses','S01 Assignment of diagnosis','VERIFIED','Participant-level diagnostic records not released'),
        ('Device family','Wacom Cintiq 16 DTK-1660 / Pro Pen 2 KP504E; HandAQUS','S01 p.5; S04','VERIFIED','Per-session serial/firmware/driver histories absent'),
        ('Calibration','Factory digitizer calibration and pen-display alignment described','S01 p.5','SUPPORTED','No calibration certificate, pressure equivalence or session logs inspected'),
        ('Operator','Individual administration by study personnel described; named visual QC evaluators','S01 Methods','SUPPORTED','No operator-to-participant/site mapping or inter-rater linkage'),
        ('Task protocol','16filename variants; local14canonical tasks collapse grade-specific dictation/copy','S01 Table3; S12 TASKS/canonical','VERIFIED','Same canonical task does not imply identical grade-specific stimuli'),
        ('Collection period','December2021–June2023; dates released as quarter/year','S01 Participants; S02 Notes','VERIFIED','Site-specific batch/protocol drift not resolved'),
        ('Timestamp/spatial units','Paper documents seconds and millimetres before JSON conversion','S01 Data Preprocessing p.5','SUPPORTED','Not independently certified clock/spatial calibration for local sessions'),
        ('Sampling cadence','Paper average 167 Hz; V3 local median-derived cadence~200/source-time-unit','S01 p.5; S13 PhaseD','CONTRADICTORY','Mean-vs-median estimators differ; reconcile distributions/clock provenance before causal inference')]
    ac=[dict(item=a,documented_evidence=b,source=c,verification_status=d,remaining_gap=e) for a,b,c,d,e in acquisition]
    v.table('ACQUISITION_PROVENANCE_MATRIX.csv',ac)
    md('ORIGINAL_DATASET_RESEARCH_VI.md',f'''# Nguồn gốc dataset và tài liệu gốc
{refs}

**VERIFIED — nguồn công bố.** Concept DOI `10.5281/zenodo.18299326` hiện trả record v2 `21236910`; API xác nhận CC BY 4.0. V2 có ZIP 1.357.028.237 bytes với MD5 công bố `e9d7a8bc5c98d1edf88824fbc90b06ca`. Không có ZIP local được đối chiếu checksum, nên không tuyên bố local là bản sao byte-identical của v2. Không tải archive hay dữ liệu cá nhân mới.

**VERIFIED — tài liệu có thể truy cập.** Đã đọc PDF data descriptor 12 trang, Zenodo HTML/API, README hai công cụ BDALab và code cohort local. `info/README.txt`, `Codebook.xlsx`, `metadata.json`, `distorted_data.xlsx` và bốn R scripts được bài báo chỉ rõ nhưng không có trong project hiện tại. Preview archive báo quá nhiều files; container endpoint403. Vì archive chứa dữ liệu trẻ em, không tải toàn bộ chỉ để lấy tài liệu. Cần chủ dữ liệu cung cấp riêng codebook/R scripts không chứa hồ sơ cá nhân. Ghi blocker, không xem mô tả bài báo là đã kiểm tra source code.

{grid(ac)}

**SUPPORTED — đối chiếu phiên bản.** Local không có trường ngày sinh/ngày ký consent theo tên cột trong hai CSV; điều này phù hợp privacy notes v2, nhưng không đủ chứng minh phiên bản hoặc kiểm tra mọi JSON/RDS. 294 released metadata IDs, 276 mô tả trong bài và 258 cohort nghiên cứu là ba tập có tiêu chí khác nhau. Không lấy số người khớp gần nhau để nhận diện cohort.

**UNVERIFIED — tài liệu bổ sung.** Không truy được codebook cấp biến, nhật ký thiết bị, bản anamnestic hoặc rule lựa chọn duplicate. Không suy ra tên thành phố từ prefix ID. Những paper downstream dùng DiaGraMo không thay thế chứng cứ tuyển mẫu của tác giả dữ liệu.
''')

    evidence=[
        dict(field='group',original_definition='HD/TD from prior anamnestic PPCC diagnosis records, including ongoing diagnosis',source='S01 p.3; S12; cleaned/raw CSV',evidence='Published definition verified; local values preserved; five specific-flag discordances',verification_status='VERIFIED',limit='Numeric 0/1 mapping SUPPORTED by local cross-tabs; author codebook and individual adjudication still absent'),
        dict(field='special_care_type_dysgraphia',original_definition='Specific special-care indicator in metadata',source='Local raw/cleaned CSV; merged cases; S01 clinical context',evidence='Binary field and relation to free-text care categories observed',verification_status='SUPPORTED',limit='Not a separately verified clinician diagnosis or standardized threshold'),
        dict(field='diagnosis',original_definition='V3 audit alias of special_care_type_dysgraphia',source='S13; scripts/data_audit_v3/phase_abc.py',evidence='Alias definition traceable in local code',verification_status='VERIFIED',limit='Do not describe as independent clinical ground truth'),
        dict(field='flag_to_omit_total',original_definition='Composite eligibility flag; user-selectable inclusion',source='S01 pp.3,6,8; cleaned CSV',evidence='Field exists in cleaned CSV, absent from raw CSV; prepare.py does not exclude it',verification_status='SUPPORTED',limit='Exact R expression/default switch not inspected; empirical OR is not a rule proof'),
        dict(field='all_group1_confirmed_dysgraphia',original_definition='Unsupported claim for every positive protected target',source='S01 ongoing-diagnosis definition; five-case read-only reconciliation',evidence='Broader HD category includes ongoing cases; specific flag differs',verification_status='CONTRADICTORY',limit='Use dataset-defined HD/group label with clinical-record provenance, not uniformly confirmed dysgraphia'),
        dict(field='label_independence_from_predictors',original_definition='Not proven at individual historical diagnosis level',source='S01 pre-existing ascertainment; S12 predictor selection',evidence='Current pipeline excludes cognitive/special-care metadata from handwriting predictors',verification_status='SUPPORTED',limit='No evidence current cohort handwriting scores generated group; historical handwriting assessment may be part of diagnosis')]
    v.table('LABEL_EVIDENCE_MATRIX.csv',evidence)
    v.json_out('LABEL_VERIFICATION_DECISION.json',dict(published_label_definition='VERIFIED',numeric_code_mapping='SUPPORTED',
        recommended_target_name='Dataset-defined handwriting-disability group; prior clinical/educational records, including ongoing assessment',
        all_group1_confirmed_dysgraphia=False,individual_case_adjudication='UNVERIFIED',specific_flag_clinically_confirmed='UNVERIFIED',
        disagreement_cases=local['disagreement_cases'],ongoing_compatible_cases=local['ongoing_cases'],
        labels_changed=False,cohort_changed=False,folds_changed=False))
    md('LABEL_PROVENANCE_VI.md',f'''# Nhãn: định nghĩa công bố và giá trị local
{refs}

**VERIFIED:** Bài gốc đã có mục Assignment of diagnosis; V3 chỉ dùng local nên thiếu bằng chứng này. Nhãn không phải một test chẩn đoán mới do nhóm dataset thực hiện. Không gọi tất cả `group=1` là confirmed Dysgraphia. Cách gọi được đề nghị: **nhóm handwriting disability theo dataset, từ hồ sơ lâm sàng/giáo dục có sẵn, bao gồm trường hợp đang đánh giá**.

**SUPPORTED:** local numeric0 tương ứng nhóm đối chứng,1 tương ứng nhóm HD rộng; codebook ánh xạ chính thức chưa được đọc. Phân biệt `special_care_type_dysgraphia` và alias `diagnosis` của V3. Tên cột không chứng minh một chẩn đoán độc lập.

{grid(evidence)}

## Năm bất đồng

Đã join từng trường hợp trên raw CSV, cleaned CSV và merged JSON: {local['all_disagreement_sources_match']}. Không phải sai lệch mới phát sinh giữa ba lớp lưu trữ ở các trường liên quan đã so sánh. Ba trường hợp có cờ ongoing: **SUPPORTED** về tương thích với định nghĩa nhóm rộng. Một trường hợp nhóm đối chứng nhưng cờ dysgraphia, và một trường hợp nhóm case với các cờ rối loạn học tập khác cần tác giả phân xử; không suy đoán động cơ gán nhãn. Giải thích hợp lý không tương đương xác minh hồ sơ cá nhân.

Bảng công khai `DISAGREEMENT_CASE_DECISIONS.csv` cho kết luận riêng từng case reference; không xuất ID, tuổi, site hay raw special-care rows. Mapping và trường liên quan nằm ở `data/provenance_verification_local/UNRESOLVED_LABEL_CASES.csv`, đã Git ignore. Tất cả năm case vẫn UNRESOLVED về adjudication cá nhân. Không sửa nhãn hoặc dùng kết quả này lựa chọn lại mô hình.

**SUPPORTED về tính độc lập:** nguồn nhãn là hồ sơ có trước; pipeline hiện tại không dùng metadata chẩn đoán/cognitive làm predictors. **UNVERIFIED:** từng người được đánh giá lịch sử bằng chính nội dung viết nào, assessors có biết các kết quả đo hay không, mức đồng nhất giữa site. Vì vậy chưa chứng minh hoàn toàn target independence theo nghĩa clinical ascertainment.
''')

    omission_rows=[]
    meanings=['Exact field definition UNVERIFIED; paper lists grade eligibility but does not map this precise variable',
        'Non-Czech language criterion; bilingual handling needs codebook',
        'Irrelevant special-care category; exact code set unknown','Missing special-care status; exact NA handling unknown']
    for field,meaning in zip(v.COMPONENTS,meanings):
        omission_rows.append(dict(field=field,meaning=meaning,source='S01 Methods/Usage Notes; cleaned CSV',
            verification_status='SUPPORTED',rule='UNVERIFIED exact original R predicate',current_pipeline='Recorded, not used for cohort exclusion'))
    omission_rows.append(dict(field='flag_to_omit_total',meaning='Composite of eligibility criteria; optional retention documented',
        source='S01 pp.3,6,8; cleaned CSV',verification_status='SUPPORTED',
        rule=f"Empirical OR mismatches={local['component_or_disagreements']}/{local['complete_component_rows']} complete rows; {local['incomplete_component_rows']} incomplete rows. Not source-code verification.",current_pipeline='3flagged protected participants retained by original eligibility expression'))
    v.table('OMISSION_RULE_PROVENANCE.csv',omission_rows)
    md('RDS_SOURCE_TRACE_VI.md',f'''# RDS và truy vết raw → cleaned
{refs}

**UNVERIFIED — RDS schema/attributes.** Hai RDS local được hash và kiểm tra dependency; decoded={rds['decoded']}. `pyreadr`={rds['readers']['pyreadr']}, `rdata`={rds['readers']['rdata']}, Rscript={rds['rscript']}. Không cài package, không thực thi R scripts làm sạch, không ghi vào nguồn. Chưa biết object type, factor levels, value labels hoặc attributes; CSV không thay thế bằng chứng đó. Dependency đề nghị nếu chủ project cho phép: pyreadr chính thức (https://github.com/ofajardo/pyreadr) trong virtualenv hoặc R chính thức; phải đọc RDS read-only và lưu schema/so sánh ở vùng local, không chạy pipeline gốc ghi đè.

**VERIFIED — nguồn thay thế sẵn có chưa được V3 khai thác:** raw CSV {local['raw_rows']}×{local['raw_columns']}, cleaned CSV {local['rows']}×{local['columns']}; cả hai có {local['raw_unique_ids']} unique IDs. Cờ tổng absent trong raw nhưng present trong cleaned. Cleaned có {local['flagged_rows']} flagged rows/{local['flagged_unique_ids']} unique IDs. Không đánh đồng số này với18/294 trong paper; kiểm tra release và RDS còn thiếu. Cohort bảo vệ có {local['protected_flagged_unique_ids']} IDs mang cờ.

**SUPPORTED — công thức:** OR của bốn component khớp cờ tổng ở {local['complete_component_rows']-local['component_or_disagreements']}/{local['complete_component_rows']} hàng complete; {local['incomplete_component_rows']} hàng không đủ dữ liệu. Đây là tính tương đương quan sát trên bảng, không xác minh code tác giả hay NA semantics. Đối chiếu flags/labels cleaned với snapshot metadata V3 của258: {local['cleaned_to_merged_cohort_all_fields_match']}; có {local['cleaned_to_merged_nonunique_ids']} IDs không unique trong nguồn CSV, được kiểm tra mọi giá trị thay vì chọn tùy ý.

**VERIFIED — tác giả cho phép lựa chọn giữ/loại:** bài gốc Usage Notes mô tả `keepOmittedParticipants_switch` (1giữ,0loại). Không có bằng chứng mặc định của script local và không tự áp dụng. Ba flagged người vẫn ở cohort vì `prepare.py` chỉ kiểm tra đủ14 task, không duplicate/invalid signal và binary `group`; không kiểm tra cờ omit. Do đó không thể gọi ba người này là vi phạm bắt buộc của tác giả chỉ từ tên cờ.

**UNVERIFIED — code nguồn:** paper nêu bốn R scripts00–03; project không chứa chúng và preview không truy được. Phải lấy bản không chứa dữ liệu cá nhân để xác minh exact formula, duplicate strategy và join cardinality.
''')
    dup=local['duplicate_summary']
    md('DUPLICATE_METADATA_INVESTIGATION.md',f'''# Duplicate metadata investigation

**VERIFIED:** {local['duplicate_count']} ID bị lặp trong raw CSV; cùng ID vẫn có hai hàng trong cleaned CSV. Xung đột gồm sáu trường HLS. Không xuất ID hoặc điểm số trong báo cáo công khai. Chứng cứ row-level và ordinal matching nằm ở `data/provenance_verification_local/DUPLICATE_METADATA_LOCAL.csv`.

{grid(dup)}

`merged_matches_raw_occurrences` là thứ tự occurrence1/2 trong các hàng cùngID, không phải số dòng file. Nếu chỉ một occurrence khớp đủ sáu trường thì xác minh được nội dung nào đã được nhúng, **không** chứng minh quy tắc first/last/clinical adjudication. Không có script merge gốc hoặc assessment timestamps để xác định đây là đánh giá lặp hay lỗi nhập. **UNVERIFIED** về lựa chọn bản ghi chính thức. Paper nói đã loại duplicate ở quy trình công bố, trong khi hai CSV local vẫn duplicate: **CONTRADICTORY**, có thể liên quan phiên bản hoặc định nghĩa duplicate; không tự chọn lời giải.

Các trường group/specific flag không xung đột giữa hai hàng theo kiểm tra trên. Trạng thái thuộc cohort phải đọc từ `protected_cohort_member` trong bảng, không suy từ việc ID có merged JSON: kiểm tra hiện tại cho thấy duplicate **không thuộc 258 người bảo vệ**. Chưa có bằng chứng trực tiếp đổi target của cohort từ hai hàng này. Nội dung HLS có thể ảnh hưởng nghiên cứu cognitive khác, nên không âm thầm deduplicate và không suy luận rằng toàn bộ feature pipeline bị đổi. Kinematic hiện tại lấy handwriting arrays, không lấy HLS làm predictor.
''')

    hypotheses=[
        ('H1','Different case/control recruitment','Published schools/PPCC referral routes; V3 city association','Not a city-level causal decomposition','Site recruitment denominators and balanced within-site sampling','SUPPORTED'),
        ('H2','Different clinical ascertainment','Routine expert diagnosis without fixed standardized cut-off; ongoing cases included','No per-site reliability data','Blinded adjudication and shared diagnostic rubric','SUPPORTED'),
        ('H3','Device/protocol source signal','V3 pressure/angle associations; session records absent','Paper describes common model; equal~200 median cadence across cities','Serial/driver/calibration logs and controlled cross-device recordings','UNVERIFIED'),
        ('H4','Demographic/educational differences','Grade/gender targeted sampling; source metadata overlap','Current sample cannot separate all pathways','Site×class×grade/age overlap and recruitment design','SUPPORTED'),
        ('H5','Preprocessing/cohort strengthens association','14 task selection and no omit filtering verified locally','Association already observed in broader V3 selection summaries; not solely a new artifact','Original cleaning code and selection graph across source cohorts','SUPPORTED'),
        ('H6','Multiple factors coexist','H1/H2/H4 compatible; source and outcome largely collinear','No identifiable percentage attribution','Crossed multi-site/device balanced cohort','SUPPORTED')]
    hyp=[dict(hypothesis=a,claim=b,evidence_supporting=c,evidence_against_or_limit=d,required_verification=e,verification_status=f,
        identifiability='Not identified causally in current cohort',source='S01;S13') for a,b,c,d,e,f in hypotheses]
    v.table('HYPOTHESIS_EVIDENCE_MATRIX.csv',hyp)
    md('SOURCE_CONFOUNDING_INTERPRETATION_VI.md',f'''# Diễn giải source–label confounding
{refs}

{grid(hyp)}

**VERIFIED (V3):** source-only Train-fold diagnostic accuracy0.972868, CramérV≈0.946827. Metadata source không được đưa trực tiếp vào Kinematic LightGBM. Mức dự đoán cao của city không chứng minh LightGBM dùng ID/city hay direct label leakage. Nó chỉ chứng minh một đường dự đoán dataset label từ source rất mạnh trong thiết kế hiện tại.

**SUPPORTED:** tài liệu mới làm H1 và H2 có nền tảng rõ hơn; recruitment category và routine diagnostic practice cần được kiểm soát. **UNVERIFIED:** bao nhiêu phần hiệu năng LightGBM/Transformer do source, motor signal, nhân khẩu hay tác động kết hợp. Bằng chứng không đủ quy kết thành phố là thiết bị hoặc cơ sở lâm sàng.

Participant-independent6-fold đo phân biệt người mới trong hỗn hợp nguồn tương tự đã thấy, không đo chuyển sang source mới. V3 `source_fold_overlap.csv` cho mọi Test city đều đã xuất hiện ở Train. City1 có hai lớp nên tính discrimination trong nguồn được, nhưng chỉ7case: mỗi lỗi làm sensitivity đổi1/7≈14.3điểm%; không phải tập xác nhận mạnh. LOCO city1 khiến Train không có lớp0; city khác khiến Test không có lớp0. Một vài split theo school/site đủ hai lớp về đại số vẫn chưa giải quyết population/device overlap hoặc power.

Nguồn ngoài phải có cả case/control trong nhiều source, đánh giá độc lập, device/operator/batch metadata và support cho từng giao điểm; không chỉ lấy thêm nhiều samples từ source độc nhất của từng lớp. Không đổi threshold hoặc chọn architecture bằng kết quả các phân tích này.
''')
    md('IDENTIFIABILITY_DECISION.md','''# Identifiability decision

**VERIFIED:** thiếu class×city overlap ở14/15city theo V3. **SUPPORTED:** recruitment có thể góp phần giải thích cấu trúc đó. **UNVERIFIED:** causal contribution của device, diagnosis, demographic và motor impairment. Không có phép adjustment đơn giản phục hồi overlap không tồn tại.

**NO-GO** cho kết luận source-invariant biomarkers, causal motor impairment hoặc clinical diagnosis từ OOF hiện tại. **CONDITIONAL GO** cho methodological analysis có công bố giới hạn, không chọn lại mô hình. Cần cohort độc lập với thiết kế crossed source×label; cần adjudication theo protocol rõ ràng trước khi kết luận về target clinical.
''')

    privacy=v.read_csv(out/'ARTIFACT_PRIVACY_CLASSIFICATION.csv')
    counts=Counter(privacy.classification)
    md('PUBLIC_DATA_RELEASE_AUDIT_VI.md',f'''# Quyền công bố và privacy
{refs}

**VERIFIED:** Zenodo API và data descriptor ghi CC BY4.0; không còn phù hợp ghi license dataset hoàn toàn unknown. **SUPPORTED:** consent mô tả chia sẻ tập biến pseudonymized cho nghiên cứu/công bố. Đây không phải bằng chứng mọi file local cũ, mọi bản gộp mới và mọi trường hợp liên kết đều được phép phát hành. Pseudonym không phải anonymization hoàn toàn.

**UNVERIFIED:** phiên bản chính xác của mọi local participant artifact và phạm vi cho derived individual OOF, SHAP, thumbnails, cognitive combinations. Công cụ pháp lý CC BY không tự chứng nhận rằng mọi liên kết có rủi ro riêng tư đã được đồng ý. Không đưa ra kết luận tư vấn pháp lý; đây là release-readiness audit.

Inventory {len(privacy)} tracked artifacts thuộc results/notebooks/docs/references; số theo phân loại: {dict(counts)}. Đây là conservative automated triage, không phải certify public-safe. Hình/binary/notebook outputs luôn cần review thủ công; không tự đánh dấu một file aggregate là an toàn chỉ bằng regex. Không scan blobs mọi commit; current tree và sự hiện diện trong HEAD đã đủ cho biết có lịch sử cần review. Không chứng nhận lịch sử sạch.

Đường dẫn ưu tiên: `results/data_audit_v3/audit_cohort.csv`, `participants_inventory.csv`, `label_disagreements.csv`, `omission_flag_details.csv`, `matched_oof_replay.csv`, `fusion_corrected_harmed.csv`; các bảng OOF trong `results/cv/`, XAI examples/waterfalls và notebooks có outputs. ID+city/site+tuổi/giới+cờ học tập có thể liên kết; individual probabilities/SHAP bổ sung suy luận mới. Không tái in hàng dữ liệu trong báo cáo này. `ARTIFACT_PRIVACY_CLASSIFICATION.csv` liệt kê đường dẫn, thay token participant trong tên file bằng `[participant-id]`; mapping đường dẫn nguyên vẹn chỉ lưu local.

**SUPPORTED — privacy v2:** ngày sinh/consent dates bị loại và administration dates làm thô. Local CSV headers phù hợp điểm này; không chứng minh toàn bộ RDS/JSON local không còn trường nhạy cảm. Cần đối chiếu release nguyên gốc có checksum và được chủ dữ liệu xác nhận trước khi tiếp tục phát hành row-level.

Không xóa file, không rewrite history, không đổi visibility và không push trong Phase 5. Kế hoạch remediation tách riêng; việc các dữ liệu đã ở GitHub không tạo thêm quyền công bố.
''')
    md('REPOSITORY_REMEDIATION_PLAN.md','''# Repository remediation plan — cần owner review

**SUPPORTED — hành động ưu tiên:** tạm dừng phát hành participant-level artifacts mới. Giữ nguyên source và lịch sử để có thể điều tra; Phase 5 chỉ thêm báo cáo aggregate và code. Vùng `data/provenance_verification_local/` đã được rule `data/` hiện có ignore, không cần sửa .gitignore toàn project.

1. Owner đối chiếu exact dataset version/checksum, CC BY attribution và consented fields với tác giả. Xác định derived OOF/SHAP có thuộc phạm vi phát hành được chấp thuận hay không.
2. Review inventory theo đường dẫn, kể cả embedded notebook/PNG/SVG và các case nhỏ có thể liên kết. Chuyển public report sang aggregate; không dùng ID hash cố định như một bảo đảm anonymization. Suppress small cells khi cần, giảm precision tuổi/địa điểm/ngày, bỏ case-level values trong public outputs. Đánh giá khả năng liên kết qua các file cũ, không chỉ từng file đơn.
3. Chỉ sau owner quyết định mới lập patch thay public exports, đưa row-level mới vào vùng ignored. Giữ evidence read-only tại máy có kiểm soát truy cập. Không thay đổi số liệu khoa học/baseline khi tạo bản aggregate.
4. Nếu phát hành trước đó vượt phạm vi được phép, xóa khỏi HEAD là chưa đủ. Cần owner phối hợp tác giả/GitHub quản lý history/forks/cache; review paths và backup trước mọi rewrite. Hiện chưa có kết luận pháp lý bắt buộc rewrite, nhưng history review là cần thiết.

Các lệnh tham khảo **chưa chạy**, chỉ phục vụ review:

```powershell
git ls-files results notebooks
git log --all -- results/data_audit_v3/audit_cohort.csv
git check-ignore data/provenance_verification_local/UNRESOLVED_LABEL_CASES.csv
```

Nếu sau này dùng công cụ history filtering, phải dùng danh sách path đã duyệt trong bản clone backup và có phê duyệt owner. Không cung cấp/chạy lệnh force-push trên working repository này. **UNVERIFIED:** phạm vi quyền cho từng derived artifact; không thể suy ra từ public download.
''')

    compat=[
        ('x/y/time','Mandatory','Units, axes and timestamps must be documented and valid','Different units require predeclared conversion; no test-label tuning'),
        ('Pen status','Mandatory','On/off-surface and transition conventions verified','Cannot reconstruct missing state without validated mapping'),
        ('Pressure','Mandatory for exact frozen7 channel pipeline','Scale and calibration compatible','Missing pressure => not exact Kinematic/Temporal/fusion validation'),
        ('Azimuth/tilt','Mandatory for exact frozen7 channel pipeline','Angle units/orientation/wrap verified','No silent zero fill'),
        ('Timestamp units/sampling','Mandatory','Actual time, jitter/gaps and native/resampled convention','Unverified clock => no physical derivative comparability'),
        ('Task identity/order','Mandatory for frozen 14-task models','Actual content, grade variants and mapping','Subset/different tasks => not exact frozen evaluation'),
        ('Task instructions','Mandatory','Fast/precise, duration, language, stimuli permissions','No name-only compatibility assumption'),
        ('Participant identity','Mandatory','Pseudonymized linkage and cohort independence','Repeated children must not cross partitions'),
        ('Participant labels','Mandatory','Pre-existing independently adjudicated or explicitly screened target','Image-level labels cannot replace participant clinical labels'),
        ('Diagnostic protocol','Mandatory','Instrument/rater/timing/ongoing status and clinical meaning','Unclear ascertainment => exploratory proxy target only'),
        ('Population','Strongly preferred','Age/grade/language/education/case-mix documented','Mismatch requires transportability caveat'),
        ('Site/device','Mandatory for source-independent claims','Both labels across multiple sources and device/operator logs','No source metadata => cannot verify source-independent validation'),
        ('License/consent','Mandatory','Access, redistribution, derived outputs and attribution reviewed','Unknown rights => local access blocked pending permission'),
        ('Cognitive scores','Optional','May contextualize labels but not enter frozen predictors','No added clinical claim from auxiliary scores'),
        ('Offline image only','Not compatible','No valid temporal/multichannel mapping','Could support separate image study only')]
    comp=[dict(requirement=a,priority=b,acceptance=c,incompatibility_or_condition=d,verification_status='SUPPORTED') for a,b,c,d in compat]
    v.table('DATASET_COMPATIBILITY_MATRIX.csv',comp)
    md('EXTERNAL_VALIDATION_REQUIREMENTS.md',f'''# External validation requirements

**SUPPORTED — prospective design requirements, chưa phải kết quả external validation.**

{grid(comp)}

## Bốn scenario

**A:** Cùng đủ7 channels và14 task, units/order/instructions được xác minh, labels/licensing đủ điều kiện: có thể đánh giá frozen Kinematic, Temporal và fixed50/50fusion sau kiểm tra schema/inference. Khóa weights, normalization, threshold và analysis plan trước nhận external outcomes. Báo cáo performance theo source, paired error overlap và uncertainty theo participant/site; không tune trên external Test.

**B:** Chỉ một số task trùng: chưa thể gọi là external validation của mô hình14 task. Chỉ dùng mô hình task-subset được phát triển/khóa trên development data trong nghiên cứu riêng được cho phép; cần đánh giá lại source Train và validation. Không chèn zero/mask rồi xem tương đương baseline.

**C:** Chỉx/y/time vàpen status: không chạy nguyên frozen7 channel pipeline như test hợp lệ. Có thể thiết kế nghiên cứu kinematic4channel riêng trước đó, nhưng phải retrain/validate trên development cohort theo protocol mới, ngoài phạm vi Phase 5. Không bịa pressure/angles.

**D:** Thiếu source metadata hoặc clinical codebook: chỉ kiểm tra technical inference khi schema đủ; chưa đủ chứng minh clinical/source-independent validation. Bổ sung quyền truy cập, ascertainment và source trước khi sử dụng làm xác nhận.

**UNVERIFIED — power:** chưa tính đủ cỡ mẫu cho external study vì chưa có mục tiêu sensitivity/specificity, margin và cấu trúc cluster chốt trước. Cần xác định endpoint, số site và số case/control trong mỗi site; tránh chọn sample size từ hiệu năng thuận lợi đã thấy.
''')
    candidates=[
        dict(candidate='Drotár & Dobeš2020 / Slovak online',source='https://doi.org/10.1038/s41598-020-78611-9',participants='120:57dysgraphia/63controls (paper Table1)',modality='Online',channels='x/y/time, pressure, azimuth/altitude, pen-state',tasks='8writing tasks; not DiaGraMo14',label_provenance='Three professionals independently assessed; paper Methods',source_diversity='Case special-needs centre vs controls school; confounding risk',license_access='Author GitHub data links verified; explicit data redistribution license UNVERIFIED',suitability='Best public online contact candidate; not plug-in frozen 14-task validation',missing='License, exact units/version, matched tasks, multi-site class overlap',status='SUPPORTED'),
        dict(candidate='Devillaine et al.2021 / Grenoble graphomotor',source='https://doi.org/10.3390/s21217026',participants='305 total,43dysgraphia; modeled43+43matched;219reference',modality='Online',channels='x/y/z/time/pressure/angles recorded; pressure/angles discarded in paper due calibration issues',tasks='6graphomotor stimuli +BHK; not same14',label_provenance='BHK specialist assessment',source_diversity='5schools and hospital; different WacomProM/ProL devices',license_access='Request-only; privacy/ethical restrictions; raw data license UNVERIFIED',suitability='Good graphomotor collaboration candidate; task/channel incompatibility remains',missing='Permission, raw schema, device correction, cohort overlap with otherFrench studies',status='SUPPORTED'),
        dict(candidate='Asselborn et al.2018 / French BHK',source='https://doi.org/10.1038/s41746-018-0049-x',participants='298:242TD/56dysgraphia',modality='Online',channels='Tablet temporal/spatial, pressure and tilt features; exact export schema UNVERIFIED',tasks='BHK copy; not14 tasks',label_provenance='BHK clinical context; independent original study',source_diversity='School/clinical recruitment; investigate overlap and source imbalance',license_access='Paper states participant data cannot be shared due ethics; not open dataset',suitability='Only potential controlled collaboration if permissions change; no currently available public external test',missing='Access approval, schema, source logs, sample independence from Grenoble',status='SUPPORTED'),
        dict(candidate='Ramlan2023/2024 Malay potential dysgraphia',source='https://data.mendeley.com/datasets/39hr8dx76p/1',participants='83children;249images (114potential/135low-potential images, not children)',modality='Offline',channels='ScannedJPG; no rawtrajectory',tasks='3Malay sentence copies; incompatible',label_provenance='Two PDM assessors; potential vs lowpotential, not confirmed clinical classes',source_diversity='Schools and Malaysian Dyslexia Association; exact per-source class support unknown',license_access='DatasetCC BY4.0 verified; paper and dataset licenses distinguished',suitability='Exclude as direct external test for current Temporal/Kinematic/fusion; offline comparator only',missing='Participant-level classcounts, source metadata, no online channels',status='VERIFIED')]
    v.table('EXTERNAL_DATASET_CANDIDATES.csv',candidates)
    md('EXTERNAL_DATASET_SHORTLIST.md',f'''# External dataset shortlist — documentation only

Ngày kiểm tra {now}. Không tải dữ liệu. Ba online study cohorts và một offline exclusion control, đều từ nguồn tác giả/publisher; không gọi đây là bốn open datasets. Các cohortFrench có thể trùng người hoặc cơ sở với nhau, chưa xác minh độc lập giữa chúng. Không dùng thông tin tuyển mẫu của các nghiên cứu này để diễn giải nguồn DiaGraMo.

{grid(candidates)}

Ưu tiên liên hệ Drotár/Dobeš về raw schema, quyền dùng và mapping task; Grenoble nếu có thể hợp tác thu14 task hoặc thiết kế task-subset riêng. BHK2018 bị restriction rõ ràng nên không có cơ sở tải hoặc giả định tiếp cận được. Malay chỉ giữ như ví dụ loại trừ do modality. **UNVERIFIED:** hiện chưa ứng viên nào đáp ứng đồng thời frozen 14-task,7 channels,clinical ascertainment và source-balanced validation.
''')
    md('VALIDATION_READINESS_DECISION.md','''# Validation readiness decision

**NO-GO** cho chạy frozen 14-task models trên các ứng viên hiện tại rồi gọi external clinical validation: chưa xác minh schema/task/license/source overlap. **CONDITIONAL GO** cho trao đổi tài liệu và thiết kế cohort mới; không tải dữ liệu, không gửi email tự động, không fine-tune.

**SUPPORTED — thứ tự ưu tiên:** (1)làm rõ nhãn/phiên bản DiaGraMo và quyền derived release; (2)xin tài liệu Drotár/Dobeš để kiểm tra khả năng hợp tác; (3)khóa endpoint, source design và task mapping trước mọi external outcomes. Nếu không có14 task, cần protocol nghiên cứu subset riêng, không âm thầm đổi baseline.
''')

    questions=[
        'Could you provide the official numeric coding of group (0/1) and the versioned codebook used for the released files?',
        'Does group=1 include all ongoing assessments, or only ongoing assessments with suspected handwriting disability? How should these cases be described?',
        'What clinical records and adjudication process determine special_care_type_dysgraphia, and how should disagreements with group be resolved without releasing identifying information?',
        'Were handwriting assessments used to establish the historical diagnosis independent of the handwriting recordings released in DiaGraMo?',
        'Could you clarify the de-identified five-case discrepancies through a secure, authorized channel and indicate whether any release erratum exists?',
        'How were cases and controls recruited in each pseudonymized city/site, and were both categories eligible at every site?',
        'Were diagnostic criteria, assessors and referral procedures comparable across schools and counselling centres?',
        'Could you supply session-level device model/serial pseudonyms, firmware/driver versions, operators and calibration/alignment records where permitted?',
        'How should the documented average 167 Hz be reconciled with approximately 0.005-second median intervals in the local SVC recordings? Were any signals resampled or timestamps quantized?',
        'Could you provide the original four R processing scripts and a codebook separately from the participant-data archive, including exact omission expressions and default switches?',
        'Do omission flags indicate optional analytic exclusions for all use cases, and what is the recommended treatment of co-occurring or non-handwriting learning difficulties?',
        'What rule selected the merged HLS record for the duplicated metadata case, and do the two rows represent repeated assessments or an input error?',
        'Could you confirm which release/checksums correspond to the 289-row cognitive CSVs and whether they differ from the 294-person records described in the paper?',
        'Does the released consent/license permit redistribution of linked participant-level metadata, OOF probabilities and SHAP explanations, and are there recommended suppression or access controls?',
        'Are additional independently recruited cohorts available with both labels at each source and a compatible 14-task, 7-channel protocol?'
    ]
    md('AUTHOR_QUESTIONS_VI.md','# Câu hỏi ưu tiên gửi tác giả/chủ dữ liệu\n\nBản nháp bằng tiếng Anh; chưa gửi. Không đính kèm ID hoặc clinical rows trong email công khai.\n\n'+'\n\n'.join(f'{i}. {q}' for i,q in enumerate(questions,1)))
    md('NEXT_RESEARCH_DECISION_VI.md','''# Quyết định nghiên cứu tiếp theo

## Case A — Provenance confirmed, vẫn confounded

**CONDITIONAL GO:** methodological comparison/ablation trong cùng cohort, có participant-independent folds cố định và giới hạn explicit. Có thể công bố empirical discrimination của dataset-defined target, cơ chế correction của fixed fusion, computational cost và nguy cơ source association. **NO-GO:** gọi superiority là clinical diagnosis/generalization, motor biomarker hoặc causal effect; không xem clarified labels là đã loại source confounding. Không cần chạy thêm architecture để giải quyết một câu hỏi về thiết kế mẫu.

## Case B — External balanced cohort available

**SUPPORTED — kế hoạch:** xác minh quyền, cohort independence, class×source overlap, schema/units/tasks và ascertainment. Khóa Kinematic/Temporal weights, Train preprocessing, thresholds và fixed50/50probability control trước nhận nhãn Test. Dùng blinded participant-level inference, paired errors và khoảng bất định, báo cáo từng source và class, tính cả specificity/calibration. Không chọn fusion weight bằng external Test. Nếu cần thích nghi task/channel thì tách development cohort/protocol mới và giữ một external lockbox độc lập.

## Case C — Không có dataset ngoài

**CONDITIONAL GO:** viết reproducibility/provenance paper hoặc methodological note về giới hạn target/source, exact OOF replay và compute trade-off; giữ mô hình hiện tại. Các phân tích city1 chỉ descriptive với7case, không làm nested search hoặc dùng uncertainty nhỏ giả tạo từ hàng triệu samples. Có thể thiết kế task-ablation hoặc source stress tests cho protocol tương lai, nhưng không tự triển khai trong Phase 5 và không đổi claim thành source independence. **NO-GO** cho clinical deployment/biomarker framing.

**Hành động ưu tiên duy nhất:** gửi tác giả gói câu hỏi về numeric target,5 discordances, versioned codebook/R scripts và quyền derived release qua owner. Nó giải quyết trực tiếp target và privacy, trong khi thêm training không thể xác định hai điều này. Đây là khuyến nghị, chưa gửi liên hệ.
''')

    blockers='''# Blockers và phạm vi chưa thực hiện

- **UNVERIFIED:** RDS chưa decode do không có pyreadr/rdata/Rscript; không cài dependency. CSV được kiểm tra riêng, không giả làm schema RDS.
- **UNVERIFIED:** original Codebook.xlsx/README.txt/metadata.json, four R scripts, distorted_data.xlsx không có local; Zenodo ZIP preview báo quá nhiều files, container403. Không tải1.4GBdataset để lấy code.
- **UNVERIFIED:** exact local-release equivalence; clinical records, numeric coding chính thức, individual adjudication và duplicate policy. Không tự sửa5labels/loại3flagged.
- **CONTRADICTORY:** local CSV duplicate còn tồn tại dù paper mô tả deduplication; average 167 Hz và median~200 khác estimators; không giải thích bằng suy đoán hardware.
- **UNVERIFIED:** session-level device/operator/calibration và causal attribution của nguồn; không tìm danh tính thật từ pseudonyms.
- **UNVERIFIED:** quyền derived participant-level releases và fullGit-history privacy; không xóa/force-push/changevisibility. Current-tree inventory là triage, không legal certification.
- **UNVERIFIED:** external raw schema/license/task matching và independence giữaFrenchcohorts. Không datasetdownload/model train/email/push/merge.

Các blocker này không cản hoàn tất báo cáo kỹ thuật với trạng thái chính xác; chúng cản đánh dấu clinical provenance fully VERIFIED hoặc external validation ready.
'''
    md('BLOCKERS.md',blockers)
    md('REPRODUCE.md','''# Tái lập Phase 5

Chạy từ repository root trên virtualenv hiện tại; không chạy prepare.py, scripts/report.py hoặc R cleaning scripts.

```powershell
.venv/Scripts/python.exe -X utf8 scripts/provenance_verification/run.py all
```

Các phase riêng: `analyze`, `reports`, `tests`, `validate`. Snapshot bảo vệ tạo một lần và giữ trong `data/provenance_verification_local/PROTECTED_BEFORE.json`; rerun không thay mốc. Code không fit model, không network-fetch, chỉ đọc source local và ghi hai folder mới dành riêng. Explicit file-path module imports ngăn va chạm report cũ.

Nguồn web đã đọc thủ công theo SOURCE_DOCUMENTS.csv; cached PDF/API/README chỉ ở vùng dataignored, không được tự fetch lại hoặc coi cached author-document hash là dataset-file hash. Registry ghi thời gian kiểm tra, khả năng truy cập và SHA khi có. PDF chính đọc bằng pypdf; PDF Asselborn dùng PyMuPDF do pypdf tokenlimit. Downloaded content chỉ gồm tài liệu/paper/metadata, không participantdataset.

Nếu copy repo sang máy khác, cần source local có quyền dùng và cached tài liệu hoặc kiểm tra lại primary URLs trước khi giữ các claim về nguồn. Evidence synthesis là bản review bằng tay gắn nguồn, không tự suy luận clinical definition từ column names. RDS absentdependency giữ blocker; nếu reader mới có sẵn phải kiểm tra schema/attributes và review report trước khi thay trạng thái.

Tests kiểm tra join không mơ hồ, prefixNA, missingness, đường dẫn output, imports, privacy guard, case decisions và RDS schema hoặc blocker. Validation hash toàn bộ tracked files tại mốc, các CSV/RDS và merged case files đã đọc, derivedcohort/features. Không quét lại4000SVC hoặc train. Tests/validation/manifests tạo kết quả riêng; manifest không self-hash. No commit/push.
''')

    main_sections=[
        ('Executive Summary','**CONDITIONAL GO** cho nghiên cứu phương pháp; **NO-GO** cho khẳng định toàn bộ case đã được chẩn đoán Dysgraphia hoặc mô hình độc lập nguồn. Báo cáo kỹ thuật hoàn tất được với blocker rõ ràng; nguồn gốc lâm sàng chưa được xác minh đầy đủ. Không huấn luyện hoặc sửa kết quả cũ.'),
        ('Original Dataset & Scientific Sources','Xem [nghiên cứu nguồn](ORIGINAL_DATASET_RESEARCH_VI.md) và [registry](SOURCE_DOCUMENTS.csv). Bài gốc đã có mục xác định nhãn và đạo đức. API nối được concept DOI với v2 và xác nhận CC BY 4.0. Chưa đối chiếu được checksum của bản dữ liệu local với archive v2; không tải archive chứa dữ liệu cá nhân.'),
        ('Data Collection and Recruitment Provenance','[Acquisition matrix](ACQUISITION_PROVENANCE_MATRIX.csv) phân biệt tài liệu về tuyển mẫu, dòng thiết bị và bằng chứng cho từng phiên. **SUPPORTED:** khác biệt tuyển mẫu có thể góp phần tạo liên hệ nguồn–nhãn. **UNVERIFIED:** tác động riêng của mỗi nguồn, thiết bị hoặc người thu thập.'),
        ('Definition and Validity of Labels','[Label provenance](LABEL_PROVENANCE_VI.md), [evidence matrix](LABEL_EVIDENCE_MATRIX.csv), [decision](LABEL_VERIFICATION_DECISION.json). Định nghĩa nhóm trong bài đã VERIFIED; ánh xạ mã số SUPPORTED; adjudication từng người UNVERIFIED. Cờ special-care không được xem là một chẩn đoán độc lập chỉ vì tên trường.'),
        ('Five Disagreement Cases',f"Đã đối chiếu {local['disagreement_cases']} trường hợp bằng cách chỉ đọc dữ liệu. Các trường liên quan khớp giữa raw CSV, cleaned CSV và merged JSON: {local['all_disagreement_sources_match']}. {local['ongoing_cases']} trường hợp tương thích với định nghĩa bao gồm ongoing diagnosis; hai trường hợp còn lại cần clarification. [Case decisions](DISAGREEMENT_CASE_DECISIONS.csv) không chứa ID hay clinical rows. Cả năm chưa được phân xử bằng hồ sơ cá nhân; không sửa nhãn."),
        ('Omission Flags and RDS Reconstruction',f"[RDS trace](RDS_SOURCE_TRACE_VI.md): decoded={rds['decoded']}. Cleaned CSV có {local['rows']} hàng × {local['columns']} cột, gồm composite và component flags. OR của bốn component có {local['component_or_disagreements']} bất đồng trên {local['complete_component_rows']} hàng đủ thông tin; không xem đây là bằng chứng về công thức R gốc. Bài báo cho phép người dùng chọn giữ/loại các trường hợp có cờ."),
        ('Duplicate Metadata and Merge Provenance','[Duplicate investigation](DUPLICATE_METADATA_INVESTIGATION.md): hai CSV local đều còn duplicate. Nội dung merged khớp occurrence thứ hai ở sáu trường HLS; điều đó không chứng minh quy tắc ưu tiên chính thức. ID này không thuộc cohort 258. Bất nhất với mô tả deduplication trong paper cần làm rõ theo phiên bản và code gốc.'),
        ('Source–Label Confounding Interpretation','[Hypothesis matrix](HYPOTHESIS_EVIDENCE_MATRIX.csv), [interpretation](SOURCE_CONFOUNDING_INTERPRETATION_VI.md), [identifiability](IDENTIFIABILITY_DECISION.md). City-only accuracy 97,29% không chứng minh LightGBM dùng trực tiếp city/ID. Tài liệu tuyển mẫu hỗ trợ H1/H2, nhưng không định lượng được phần hiệu năng do nguồn hay vận động.'),
        ('Acquisition & Protocol Unknowns','Paper mô tả Cintiq 16/Pro Pen 2, factory calibration, alignment và đơn vị seconds/mm. **SUPPORTED** ở mức tài liệu, chưa có log từng phiên. Average 167 Hz và cadence suy từ median khoảng 200 trong V3 là hai phép đo khác nhau, cần đối chiếu trước khi kết luận mâu thuẫn vật lý. 16 biến thể tên file được quy về 14 tác vụ do dictation/copy phụ thuộc khối lớp.'),
        ('Public Data Release and Privacy Findings','[Release audit](PUBLIC_DATA_RELEASE_AUDIT_VI.md) và [remediation](REPOSITORY_REMEDIATION_PLAN.md) nêu các đường dẫn cần review. Giấy phép CC BY 4.0 được xác minh; phiên bản local và quyền phát hành derived OOF/SHAP chưa được xác minh đầy đủ. Inventory công khai che ID trong tên file; inventory nguyên vẹn giữ local. Không xóa hoặc sửa lịch sử Git.'),
        ('External Validation Feasibility','[Requirements](EXTERNAL_VALIDATION_REQUIREMENTS.md), [shortlist](EXTERNAL_DATASET_SHORTLIST.md) có ba cohort online từ nghiên cứu gốc và một đối chứng loại trừ offline. Ưu tiên liên hệ tác giả cohort Slovak về schema/quyền dùng. Các cohort Pháp có hạn chế access, tác vụ và thiết bị. Chưa ứng viên nào đủ điều kiện xác nhận nguyên mô hình 14 tác vụ, 7 kênh.'),
        ('Scientific Implications for LightGBM and Stroke Transformer','**VERIFIED theo V3:** Macro-F1 Kinematic khoảng 0,8472; Temporal 0,7132; fixed fusion 0,8407 trên target group. Việc làm rõ định nghĩa nhãn không biến các số này thành hiệu năng chẩn đoán đã xác nhận độc lập. Không chứng minh LightGBM chỉ học nguồn, cũng không chứng minh Transformer học motor biomarkers.'),
        ('Verified vs Unverified Evidence','Bốn mức VERIFIED/SUPPORTED/UNVERIFIED/CONTRADICTORY áp dụng theo đúng phạm vi claim. Định nghĩa công bố không thay thế adjudication từng người; mô tả code không thay thế kiểm tra source code; giấy phép không tự chứng minh mọi liên kết dữ liệu đều được chấp thuận. Xem các matrix, decision JSON và blockers.'),
        ('Required Author/Owner Clarifications','[15 câu hỏi tiếng Anh](AUTHOR_QUESTIONS_VI.md) ưu tiên mã hóa nhãn, ongoing cases, năm bất đồng, R scripts, omission/duplicate policy, phiên bản và quyền phát hành. Chưa gửi liên hệ; không đưa ID hoặc clinical rows vào trao đổi công khai.'),
        ('Research GO / CONDITIONAL GO / NO-GO Decision','[Next research decision](NEXT_RESEARCH_DECISION_VI.md): CONDITIONAL GO cho nghiên cứu phương pháp và tái lập có giới hạn minh bạch; NO-GO cho triển khai lâm sàng hoặc claim độc lập nguồn. Ưu tiên xin codebook, R scripts theo phiên bản và clarification nhãn, trước khi nghiên cứu thêm kiến trúc.'),
        ('Reproducibility and Limitations','[Reproduce](REPRODUCE.md), [validation](VALIDATION_CHECKS.json), [manifest](ARTIFACT_MANIFEST.json), [blockers](BLOCKERS.md). Kiểm tra hash toàn bộ tracked files tại mốc và các protected inputs đã chọn. Không tái chạy sensor audit V3, không train, không tải dataset, không push. Chưa quét toàn bộ Git history về privacy; RDS vẫn thiếu reader.')]
    md('DATASET_PROVENANCE_VERIFICATION_VI.md','# Phase 5 — Dataset Provenance & Clinical Label Verification\n\n'+refs+'\n\n'+'\n\n'.join(f'## {i}. {title}\n\n{text}' for i,(title,text) in enumerate(main_sections,1)))
