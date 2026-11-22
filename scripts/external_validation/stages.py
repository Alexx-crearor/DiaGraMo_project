"""Research stages and actual dataset evidence (never infer trajectories from features)."""
import json
from pathlib import Path
import time


PAPER='https://doi.org/10.1038/s41598-020-78611-9'
SUPPLEMENT='https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-020-78611-9/MediaObjects/41598_2020_78611_MOESM1_ESM.pdf'


def acquire(c,reader,budget):
    start=time.monotonic();budget.admit(300)
    if not (c.OUT/'phase0_status.json').exists():raise ValueError('Phase0 required')
    # Pin public files to the commit independently inspected with the GitHub connector.
    files=[('README.md',393,'608c3f7794c46a247099d5d09ea273ec88f7f1d6'),
        ('data2_SciRep_pub.xlsx',17201,'2f709020eef333792099f6e71e146a2eac3e3695'),
        ('dataSciRep_public.zip',16731844,'95d17ffe91521d1efe4a8592c6105156a07341a7')]
    checks=[]
    for name,size,blob in files:
        url=f'https://raw.githubusercontent.com/{c.REPO}/{c.COMMIT}/{name}'
        checks.append(dict(name=name,**reader.download(url,c.LOCAL/name,32*1024*1024,blob,size)))
    checksum_record=dict(repository=c.REPO,commit=c.COMMIT,checked_utc=c.stamp(),files=checks,
        access='Explicit user-authorized bounded read-only download from author public repository linked by paper; redistribution not authorized',
        license='No standalone data license found in three-file repository; do not infer article CC BY applies to data')
    if (c.OUT/'external_dataset_checksums.json').exists():
        old=c.load('external_dataset_checksums.json')
        if old['files']!=checks or old['commit']!=c.COMMIT:raise ValueError('Pinned source changed')
    else:c.write('external_dataset_checksums.json',checksum_record)
    inventory=reader.extract_safe(c.LOCAL/'dataSciRep_public.zip',c.LOCAL/'extracted')
    c.table('ZIP_INVENTORY_LOCAL.csv',inventory,local=True)
    sheets=reader.xlsx_cells(c.LOCAL/'data2_SciRep_pub.xlsx')
    c.write('EXCEL_CELLS_LOCAL.json',sheets,local=True)
    print('ZIP member count',len(inventory),'total bytes',sum(r['bytes'] for r in inventory),flush=True)
    print('ZIP suffix counts',__import__('collections').Counter(Path(x['name']).suffix.lower() for x in inventory if not x['directory']),flush=True)
    print('ZIP top-level summaries',sorted(set(x['name'].split('/')[0] for x in inventory)),flush=True)
    print('Excel sheets/cell counts',[(k,len(v)) for k,v in sheets.items()],flush=True)
    c.write('ACQUISITION_STATE.json',dict(commit=c.COMMIT,downloaded_files=len(checks),zip_members=len(inventory),
        uncompressed_bytes=sum(x['bytes'] for x in inventory),sheets=list(sheets),downloaded_raw_redistributed=False))
    # Classification and validation will be finalized after actual member schema inspection.
    c.status(1,'PARTIAL',start,dict(downloads_verified=checks,zip_members=len(inventory)),
        dict(git_blobs_match=True,zip_crc_checked=True,safe_extraction=True,excel_read_without_formula_execution=True),
        ['external_dataset_checksums.json','ACQUISITION_STATE.json'],reason='Awaiting semantic inventory of actual archive members')


def inventory_and_audit(c,reader,budget):
    import collections
    import re
    import numpy as np
    import pandas as pd
    import xml.etree.ElementTree as ET
    from sklearn.model_selection import StratifiedKFold
    start=time.monotonic();budget.admit(120)
    sheets=reader.xlsx_cells(c.LOCAL/'data2_SciRep_pub.xlsx')
    participants=reader.workbook_participants(sheets)
    participants=participants.sort_values('participant_id').reset_index(drop=True)
    base=c.LOCAL/'extracted/dataSciRep_public'
    recording_rows=[];channel_rows=[];sample_total=0;source_hashes=[]
    for p in sorted(base.rglob('*.svc')):
        match=re.fullmatch(r'u(\d{5})s(\d{5})_hw(\d{5})\.svc',p.name)
        if not match:raise ValueError('Unrecognized source recording filename')
        pid,session,record=match.groups()
        if pid not in set(participants.participant_id):raise ValueError('Recording has no unique workbook label')
        # Parent directory must independently agree with filename identity.
        if p.parent.name!='session'+session or p.parent.parent.name!='user'+pid:raise ValueError('Path/filename ID mismatch')
        source_hashes.append(dict(path=p.relative_to(c.LOCAL).as_posix(),sha256=c.sha(p),bytes=p.stat().st_size))
        try:
            x=reader.load_svc_numeric(p)
        except ValueError as error:
            recording_rows.append(dict(participant_id=pid,session_id=session,recording_id=record,path=p.relative_to(c.LOCAL).as_posix(),
                samples=0,columns=None,header_matches=False,nonfinite=None,valid_signal=False,
                reader_error=str(error),candidate_pen_binary=None,task_boundary_annotations=False,sha256=c.sha(p)))
            continue
        dt=np.diff(x[:,2]);positive=dt[dt>0]
        candidate_pen=x[:,3];binary=set(np.unique(candidate_pen))<={0,1}
        sample_total+=len(x)
        recording_rows.append(dict(participant_id=pid,session_id=session,recording_id=record,path=p.relative_to(c.LOCAL).as_posix(),
            samples=len(x),columns=x.shape[1],header_matches=True,valid_signal=True,reader_error='',nonfinite=int((~np.isfinite(x)).sum()),
            candidate_time_column=2,candidate_pen_column=3,semantic_status='SUPPORTED by pattern; primary export order/units UNVERIFIED',
            dt_zero=int((dt==0).sum()),dt_negative=int((dt<0).sum()),dt_positive=int((dt>0).sum()),
            dt_min_positive=float(positive.min()) if len(positive) else None,
            dt_median_positive=float(np.median(positive)) if len(positive) else None,
            dt_p95_positive=float(np.quantile(positive,.95)) if len(positive) else None,
            duration_native=float(x[-1,2]-x[0,2]),candidate_pen_binary=binary,
            candidate_pen_transitions=int((candidate_pen[1:]!=candidate_pen[:-1]).sum()),
            candidate_pen_down_fraction=float(np.mean(candidate_pen==1)),
            candidate_pressure_zero_when_up_fraction=float(np.mean(x[candidate_pen==0,6]==0)) if np.any(candidate_pen==0) else None,
            negative_candidate_pressure=int((x[:,6]<0).sum()),
            candidate_gaps_over_10x_recording_median=int((dt>10*np.median(positive)).sum()) if len(positive) else None,
            task_boundary_annotations=False,sha256=c.sha(p)))
        for col in range(7):
            a=x[:,col]
            channel_rows.append(dict(participant_id=pid,session_id=session,recording_id=record,column=f'c{col}',
                mean=float(a.mean()),std=float(a.std()),minimum=float(a.min()),median=float(np.median(a)),
                p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)),maximum=float(a.max()),samples=len(a)))
    recordings=pd.DataFrame(recording_rows)
    assert len(recordings) and set(recordings.participant_id)==set(participants.participant_id)
    counts=recordings.groupby('participant_id').size()
    participants['recording_count']=participants.participant_id.map(counts)
    participants['candidate_fold']=-1
    assignments=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(participants.participant_id,participants.label)):
        train=set(participants.iloc[tr].participant_id);test=set(participants.iloc[te].participant_id)
        assert not train&test and len(set(participants.iloc[tr].label))==2 and len(set(participants.iloc[te].label))==2
        participants.loc[te,'candidate_fold']=fold
        assignments.append(dict(fold=fold,train_participants=len(tr),test_participants=len(te),
            train_class0=int((participants.iloc[tr].label==0).sum()),train_class1=int((participants.iloc[tr].label==1).sum()),
            test_class0=int((participants.iloc[te].label==0).sum()),test_class1=int((participants.iloc[te].label==1).sum()),overlap=0))
    recordings['candidate_fold']=recordings.participant_id.map(participants.set_index('participant_id').candidate_fold)
    assert recordings.groupby('participant_id').candidate_fold.nunique().max()==1
    c.table('external_participant_inventory.csv',participants,local=True)
    c.table('external_recording_inventory.csv',recordings,local=True)
    c.table('CHANNEL_STATS_LOCAL.csv',channel_rows,local=True)
    c.write('RAW_RECORDING_CHECKSUMS.json',dict(files=source_hashes),local=True)
    c.write('EXCEL_CELLS_LOCAL.json',sheets,local=True)
    # XML is not a label source and contains free-text history/comments; never export those strings.
    xml_summary=[]
    for p in sorted(base.glob('*.xml')):
        root=ET.fromstring(p.read_bytes());users=root.findall('user')
        ids=[e.findtext('id') for e in users]
        fields=sorted(set(e.tag for u in users for e in u))
        labeled=[u for u in users if u.findtext('id') in set(participants.participant_id)]
        dupcounts={}
        for key in ['history','comments']:
            values=[(u.findtext(key) or '').strip() for u in labeled]
            count=collections.Counter(x for x in values if x)
            dupcounts[key+'_duplicate_nonempty_groups']=sum(n>1 for n in count.values())
        xml_summary.append(dict(file=p.name,rows=len(users),unique_ids=len(set(ids)),matched_labeled_ids=len(set(ids)&set(participants.participant_id)),
            columns=fields,declared_encoding='ISO-8859-1 (observed declaration; no semantic text recoding used)',
            individual_source_metadata_available=False,contains_free_text=True,**dupcounts))
    c.write('XML_SCHEMA_SUMMARY.json',xml_summary)
    # Summaries first average recordings within participant; independent summary unit remains participant.
    channels=pd.DataFrame(channel_rows).groupby(['participant_id','column'],as_index=False).mean(numeric_only=True)
    summaries=[]
    for col,g in channels.groupby('column'):
        summaries.append(dict(column=col,n_participants=len(g),units='native values; physical units UNVERIFIED',
            mean_of_participant_means=float(g['mean'].mean()),median_of_participant_medians=float(g['median'].median()),
            min_of_participant_minima=float(g.minimum.min()),max_of_participant_maxima=float(g.maximum.max()),
            semantic_mapping='UNVERIFIED exact export schema; no conversion applied'))
    c.table('channel_statistics.csv',summaries)
    person=recordings.groupby('participant_id').mean(numeric_only=True)
    c.table('timestamp_quality.csv',[dict(unit='participant-mean of recording statistics; candidate c2 only',
        participant_count=len(person),recordings=len(recordings),samples=sample_total,
        median_candidate_dt_native=float(person.dt_median_positive.median()),
        zero_intervals=int(recordings.dt_zero.sum()),negative_intervals=int(recordings.dt_negative.sum()),
        gaps_10x_median=int(recordings.candidate_gaps_over_10x_recording_median.sum()),
        timestamp_unit='UNVERIFIED; no conversion to seconds/Hz',threshold='10x own-recording positive-dt median; audit diagnostic only')])
    c.write('participant_split_integrity.json',dict(unique_workbook_ids=len(participants),unique_svc_ids=recordings.participant_id.nunique(),
        labeled_recordings=len(recordings),label_classes=sorted(map(int,participants.label.unique())),
        source_label_counts={str(k):int(val) for k,val in participants.label.value_counts().items()},
        repeated_recordings_participants=int((participants.recording_count>1).sum()),
        candidate_split='Stratified5 participant IDs seed42; not a source-held-out split',folds=assignments,
        all_recordings_of_person_same_fold=True,physical_identity_independently_verified=False,
        note='Source ID grouping verified; shared/nonempty XML history/comments among matched cases checked but not an identity guarantee'))
    c.table('source_label_contingency.csv',[dict(source='UNAVAILABLE at participant level',class0=int((participants.label==0).sum()),
        class1=int((participants.label==1).sum()),source_independent_evaluation='UNVERIFIED',
        source_definition='Paper describes case clinical centre/control school routes; cannot assign source labels from outcome without circularity')])
    file_inventory=[]
    for suffix,n in collections.Counter(p.suffix or '(no extension)' for p in base.rglob('*') if p.is_file() and not p.name.endswith('.decoded_local.txt')).items():
        file_inventory.append(dict(location='ZIP',file_type=suffix,file_count=n,detail='Exact source paths/hashes retained local-only'))
    c.table('external_file_inventory.csv',file_inventory)
    schema=[dict(resource='SVC',schema='header sample count;7numeric columns;no column names',rows=sample_total,columns=7,
        identity='path user/session + filename IDs checked',physical_units='UNVERIFIED'),
        dict(resource='Excel',schema='ID;diag;sex;hand;age; extra sparse unlabeled cells',rows=len(participants),columns=5,
        identity='Mixed text5digit/numeric3digit IDs canonicalized to5digits; collisions rejected and all120SVC IDs matched',physical_units='Age meaning documented by header; recording units absent')]
    for x in xml_summary:schema.append(dict(resource=x['file'],schema=';'.join(x['columns']),rows=x['rows'],columns=len(x['columns']),identity='XML contains additional non-study IDs',physical_units='No sensor unit dictionary'))
    c.table('external_schema_inventory.csv',schema)
    c.table('external_label_dictionary.csv',[
        dict(field='diag',source_value='DYSGR',analysis_value=1,count=int((participants.label==1).sum()),definition='Study dysgraphia group',evidence='Excel + paper Methods; existing assessment by3professionals'),
        dict(field='diag',source_value='0',analysis_value=0,count=int((participants.label==0).sum()),definition='Study non-dysgraphia group',evidence='Excel + paper Table1/Methods; do not infer clinical equivalence to DiaGraMo group')])
    inv=dict(classification='A',raw_trajectories_available=True,engineered_feature_tables_available=False,
        participants=len(participants),class0=int((participants.label==0).sum()),class1=int((participants.label==1).sum()),
        svc_recordings=len(recordings),samples=sample_total,columns=7,
        valid_recordings=int(recordings.valid_signal.sum()),invalid_or_empty_recordings=int((~recordings.valid_signal).sum()),
        participants_with_valid_signal=int(recordings.loc[recordings.valid_signal,'participant_id'].nunique()),
        paper_task_conditions=8,actual_task_annotations=0,recording_to_task_mapping='UNKNOWN; one hw code for119IDs, two for1ID; no 8task boundary annotations',
        header_and_numeric_checks_passed=bool(recordings.header_matches.all() and recordings.nonfinite.sum()==0),
        source_metadata_available=False,physical_units_verified=False,column_order_verified=False,
        candidate_pen_all_binary=bool(recordings.candidate_pen_binary.all()),
        label_formulas_verified=int(participants.label_formula_present.sum()),
        excel_id_padding_applied=int(participants.id_padding_applied.sum()),
        median_dt_candidate_native=float(person.dt_median_positive.median()),
        extra_xml_participants=[x['rows'] for x in xml_summary],
        zero_intervals=int(recordings.dt_zero.sum()),negative_intervals=int(recordings.dt_negative.sum()))
    c.write('DATASET_INVENTORY.json',inv)
    c.md('PHASE1_DATASET_PROVENANCE.md',f'''# Phase 1 — Nguồn và dữ liệu thực tế

Nguồn chính: [{PAPER}]({PAPER}); original repo `{c.REPO}`, commit `{c.COMMIT}`. Methods mô tả120trẻ, ba chuyên gia đánh giá, tuyển case tại Centre for Special-Needs Education và đối chứng tại elementary school; Wacom Intuos Pro Large,8điều kiện viết. Supplement1trang đã tải và xem hình template gốc. Không suy những quy trình này từ paper khác.

Đã tải có giới hạn đúng3publicfiles và xác minh Git blob SHA1 theo commit, SHA256/kích thước riêng. ZIP365members,124files (121SVC,2XML,1.DS_Store),81.422.448bytes giải nén theo inventory, không execute code. `external_dataset_checksums.json` lưu nguồn cụ thể. Connector GitHub chỉ nhận UTF8 nên binary được tải trực tiếp từ raw.githubusercontent.com tại SHA đã khóa. Không upload lại source.

Phân loại **A — có raw online trajectories**, không phải engineered-feature dataset. Có {inv['participants']} IDs/{inv['svc_recordings']} SVC; {inv['valid_recordings']} recording hợp lệ, {inv['invalid_or_empty_recordings']} rỗng/không hợp lệ; tổng {inv['samples']} samples trong recording hợp lệ. Cả {inv['participants_with_valid_signal']} người vẫn có tín hiệu hợp lệ. Excel có 120 IDs,57 DYSGR/63 zero; header ID/diag/sex/hand/age. 49 ID dạng số có ba chữ số được chuẩn hóa zero-padding thành năm chữ số giống filename, độc lập nhãn, không collision và join đủ120; ID gốc chỉ giữ local. Một cached label formula tham chiếu đơn trong cùng cột đã được đối chiếu, không tính công thức hay mở macro. Các ô trống/formatting ngoài bảng không tạo participant mới. Hai XML có IDs thử nghiệm/ngoài cohort, không lấy chúng làm120nhãn.

Không tìm LICENSE trong listing3files hoặc schema/units dictionary. Public repository và paper cho phép tiếp cận dữ liệu nghiên cứu; giấy phép redistribution và derivative release chưa xác minh. Chỉ đọc local theo yêu cầu người dùng; XML free-text history/comments là dữ liệu cần bảo vệ, không xuất publicrows.

Số8là điều kiện trong paper/template, **không phải8recordings/người đã xác minh**. Ranh giới nhiệm vụ và semantics tuyệt đối của cột không có trong files. Issue2 của originalrepo hỏi đúng7cột không header và không có câu trả lời; đây là blocker tài liệu, không phải xác nhận giả thuyết của người hỏi.
''')
    previous=c.load('phase1_status.json')
    previous.update(status='COMPLETE',classification='A',semantic_limitations=['Exact units','Column mapping','Task boundaries'],
        inventory=inv,reason=None,utc=c.stamp(),artifacts=['PHASE1_DATASET_PROVENANCE.md','external_file_inventory.csv',
            'external_schema_inventory.csv','external_label_dictionary.csv','external_dataset_checksums.json','DATASET_INVENTORY.json'])
    c.write('phase1_status.json',previous)
    c.md('PHASE2_SIGNAL_AUDIT.md',f'''# Phase 2 — Audit raw và đơn vị độc lập

Đã khảo sát toàn bộ {len(recordings)} SVC. {inv['valid_recordings']} recording qua kiểm tra header/7columns/finite values; {inv['invalid_or_empty_recordings']} file rỗng bị reader từ chối. Vì vậy all-files-pass={inv['header_and_numeric_checks_passed']}, không giả đánh dấu toàn bộ là hợp lệ. Một người có file rỗng và một recording hợp lệ khác, nên vẫn có120người với tín hiệu. Không đổi source hay tự xóa bản ghi. File không có fieldnames. Phân tích theo c0…c6 và gọi timestamp/pen-state **candidate**, không tự gán physical units.

Candidate c2 có median positiveΔt={inv['median_dt_candidate_native']} native units, zeroΔt={inv['zero_intervals']}, negativeΔt={inv['negative_intervals']}; candidate c3 binary ở mọi recording={inv['candidate_pen_all_binary']}. Khoảng nhảy>10×median riêngrecording chỉ là diagnostic, không chia hay resample dữ liệu. Áp suất/góc chỉ audit giá trị số/vùng theo cột; không chuyển altitude→tilt hoặc ticks→seconds.

Excel–SVC join đủ120ID, labelsbinary, một IDcó2recordings. Đã tạo candidate5folds seed42, mọi recordingscùngngười cùngfold; không có sample/window randomsplit. Đây là grouping theo IDnguồn, không bằng chứng độc lập vật lý tuyệt đối. XML history/comments có thể nhạy cảm; chỉ kiểm tra trùng nonempty trong phần join được, không publicnội dung.

Các bảng recording/participant và candidatefolds chỉ ở `data/external_datasets/drotar_dobes_2020/`. Publicchannelstatistics trung bình recordingtrongngười trước khi tổng hợp; không coi hàng triệu samples là replicates. Không có city/site/device serial/date theo participant, không tạo source bằng cách sao label sang source. Sourceindependence vẫn UNVERIFIED. Metadataage/sex/hand không dùng làm predictors.

**PARTIAL:** structural/grouping audit hoàn tất, nhưng exactschema/units và taskboundaries chưa xác minh, nên chưa đủ cho physical harmonization hay clinical/raw-model comparison.
''')
    c.status(2,'PARTIAL',start,inv,dict(all_headers_and_finites_pass=inv['header_and_numeric_checks_passed'],
        participant_join_complete=True,repeated_recordings_grouped=True,physical_units_verified=False),
        ['PHASE2_SIGNAL_AUDIT.md','channel_statistics.csv','timestamp_quality.csv','participant_split_integrity.json','source_label_contingency.csv'],
        reason='Exact export column definitions/units and task boundaries absent; source-level metadata absent')
    return inv


def compatibility(c,inv):
    start=time.monotonic()
    dia=['RCFT-recognition','RCFT-recall','dictation','big-spiral-fast','small-spiral-fast','saw-fast',
        'upper-loops-fast','lower-loops-fast','rainbow-fast','combined-loops-fast','small-spiral-precise','lower-loops-precise','combined-loops-precise','copy']
    slovak=[('letter_l_normal','l','normal'),('letter_l_fast','l','fast'),('syllable_le_normal','le','normal'),
        ('syllable_le_fast','le','fast'),('word_leto','leto','unspecified'),('pseudoword_lamoken','lamoken','unspecified'),
        ('word_diacritics','hračkárstvo','unspecified'),('sentence','V lete bude teplo a sucho','unspecified')]
    pairs=[]
    for a in dia:
        for b,content,speed in slovak:
            category='INCOMPATIBLE';reason='Different geometric/linguistic content and motor demand; no task-level recording labels'
            if a=='copy':
                category='CLOSE MOTOR ANALOGUE' if b=='sentence' else 'WEAK ANALOGUE'
                reason='Visible text-production analogue; different language/stimulus/grade and no external task boundaries; not exact transfer'
            elif a=='dictation':
                category='WEAK ANALOGUE'
                reason='Both linguistic handwriting, but auditory dictation vs provided Slovak writing template differ in production demands'
            elif 'loops' in a and (b.startswith('letter') or b.startswith('syllable')):
                category='WEAK ANALOGUE'
                reason='Repeated curved strokes may overlap, but graphomotor loops are not letters/syllables; direction/instructions differ'
            pairs.append(dict(diagramo_task=a,slovak_reported_task=b,category=category,justification=reason,
                slovak_content=content,slovak_speed=speed,diagramo_speed='fast' if 'fast' in a else ('precise' if 'precise' in a else 'not speed-matched'),
                motor_complexity='Task-dependent; not matched quantitatively',pen_lifts='Not equated; sequence and instructions differ',
                task_duration='Not matched; boundaries unavailable',language='Czech vs Slovak; geometric tasks nonlinguistic',
                age_grade_effects='DiaGraMo grade3/4 vs Slovak ages8–15; do not assume equivalent maturation',
                mapping_executable=False,evidence=PAPER+' Methods + Supplement; DiaGraMo paper Table3 + prepare.py'))
    c.table('TASK_COMPATIBILITY_MATRIX.csv',pairs)
    candidates=['x','y','timestamp','pen_state','azimuth_candidate','altitude_candidate','pressure_candidate']
    channels=[]
    for i,name in enumerate(candidates):
        channels.append(dict(external_column=f'c{i}',candidate_meaning=name,
            diagramo_channel=['x','y','time','pen_status','azimuth','tilt','pressure'][i],
            classification='Unknown',evidence_level='SUPPORTED candidate by numerical pattern; exact exporter schema UNVERIFIED',
            observed_support='Allfiles binary0/1' if i==3 else ('Mostly positive increasing integer ticks' if i==2 else 'Ranges consistent with candidate device channel; no primary schema'),
            conversion='NONE; do not infer ticks/ms, coordinate scaling, angle order or altitude-to-tilt',
            units='UNVERIFIED',source='Downloaded SVC no header; original paper Methods/Table2; original repo issue2 unanswered'))
    c.table('CHANNEL_COMPATIBILITY_MATRIX.csv',channels)
    decisions=dict(dataset_classification='A',raw_available=True,outcome='E for signal-model execution under current evidence; label-only control separately admitted',
        A=dict(status='NOT VALID',run=False,reason='No exact14task contract, absent task boundaries and unverified physical/export schema'),
        B=dict(status='NOT FEASIBLE',run=False,reason='Only semantic analogues; no exact shared task/recording mapping or development subset'),
        C=dict(status='BLOCKED',run=False,reason='Shared channel order, timestamp/angle units and meanings unverified; no approved harmonization'),
        D=dict(status='PARTIAL: PRIOR-ONLY CONTROL ADMITTED',run_prior=True,run_signal_models=False,
            reason='120unique labeled IDs support5fold label-only control; actual Kinematic/Temporal benchmark waits for exact export schema. This is not frozen external validation'),
        no_zero_filled_channels=True,no_assumed_task_indices=True,no_altitude_conversion=True,
        gate_basis='Source documentation and actual files; no external-Test model scores inspected',locked_before_scores=True)
    c.write('TRANSFER_PROTOCOL_DECISION.json',decisions)
    c.md('PHASE3_COMPATIBILITY_REPORT.md','''# Phase 3 — Task/channel compatibility và chọn đường hợp lệ

Taskmatrix xét đủ14×8=112cặp theo nội dung, vận động, speed, penlifts, duration, instructions, grade/language. Không có EXACT MATCH. Copy–sentence chỉ CLOSE MOTOR ANALOGUE, không phải mapping thực thi; không có taskboundaryđể tách8điều kiện từSVC. Không gán thứ tự task theo vùngtrênpage hay chia đều samples.

Channelmatrix phân biệt candidate meaning với exactexport-schema. Có7cột thật nhưng không có header/dictionary nguồn; c2có patternclock, c3binary, c6zero thường khi c3zero hỗ trợ giả thuyết, không xác minh đơnvị. Không dùng parserDiaGraMo vì thứ tự pressure/góc/timestamp khác candidate; không giả milliseconds hoặc mmscale. Issue2 chínhrepo hỏi7columns, không authoranswer. Generic SVC parser hoặc paperdownstream không thay sourceexportcontract.

A NOT VALID: task/channel contract sai; không load frozenweights. B NOT FEASIBLE: analogue khác sharedtaskmapping. C BLOCKED: sharedrawsemantics chưa verified. D chỉ **prior-only labelbalancecontrol** hợp lệ ngay:120IDs và labels join đủ,5foldsparticipant, không phụ thuộc units. Handcrafted/Temporal học từraw củaD vẫnBLOCKED; không lấydata-indexfeatures hoặc nativechannelguess để lách mandatoryunit/schema gate. Datasetrawcóthật, không gọi nó feature-only/unusabletoàn bộ.

Đây là OutcomeE đối với câu hỏi học mô hình tín hiệu hiện tại; không phải tuyên bố dataset không thể dùng sau clarification. Không chọn path bằng expectedF1; không có scoreKinematic/Temporalngoài nào được tính.
''')
    c.status(3,'COMPLETE',start,dict(pairs=len(pairs),exact_matches=0,decisions=decisions),
        dict(no_semantic_contract_fabricated=True),['PHASE3_COMPATIBILITY_REPORT.md','TASK_COMPATIBILITY_MATRIX.csv','CHANNEL_COMPATIBILITY_MATRIX.csv','TRANSFER_PROTOCOL_DECISION.json'])
    return decisions


def harmonize(c,reader,inv):
    start=time.monotonic()
    config=dict(mode='Read-only numeric schema adapter; signal-model conversion disabled',
        input='Count header +7native numeric columns',output='Unchanged float64 array in acquisition order',
        columns_named=['c0','c1','c2','c3','c4','c5','c6'],physical_mapping='UNVERIFIED',
        resampling=False,time_unit_conversion=False,angle_conversion=False,missing_channel_imputation=False,
        fitted_normalization=False,task_mapping=False,labels_in_input_tensors=False,
        raw_model_training_admitted=False,prior_control_uses_only_train_class_counts=True)
    c.write('adapter_configuration.json',config)
    checks=[]
    passed=0;rejected=0
    for p in sorted((c.LOCAL/'extracted').rglob('*.svc')):
        try:a=reader.load_svc_numeric(p);b=reader.load_svc_numeric(p)
        except ValueError:
            rejected+=1
            continue
        import numpy as np
        assert np.array_equal(a,b)
        passed+=1
    for item,status,scope in [('deterministic_reader','PASS',f'{passed}valid SVC reread identical; {rejected}invalid/empty rejected'),('sample_preservation','PASS','Valid array count matches header; no cropping/normalization'),
        ('no_fake_channels','PASS','Only source7columns returned'),('no_label_input','PASS','Reader API accepts path only; no label argument'),
        ('task_mapping','NOT_RUN','No exact mapping'),('time_or_angle_conversion','NOT_RUN','No verified unit transform'),
        ('window_boundary_masks','NOT_RUN','No signal-model windows generated'),('train_only_scaler','NOT_APPLICABLE','No signal model or scaler fit')]:
        checks.append(dict(check=item,status=status,scope=scope))
    c.table('harmonization_checks.csv',checks)
    c.write('preprocessing_validation.json',dict(numeric_reader_passed=True,physical_harmonization_completed=False,
        signal_windows_generated=0,scalers_fit=0,labels_in_signal_arrays=False,terminal_samples_preserved=True,
        window_mask_semantics='NOT_RUN; no false passing assertion on unavailable pipeline'))
    c.md('PHASE4_HARMONIZATION_REPORT.md','''# Phase 4 — Adapter có kiểm soát

Đã thực thi reader độc lập trên121SVC:120recording có tín hiệu trả đúng7cột/hàng theoheader và reread bit-identical; một file0mẫu bị từ chối. Không đổi thứ tự hoặc mất terminal sample của các recording hợp lệ. Không gọi hoặc sửa preprocessing DiaGraMo. API không nhận label nên label không vào array.

**PARTIAL:** chỉ hoàn thành numericadapter. Physicalharmonization, semantic taskmapping, windowing/masks, normalizer và CNNinput chưa thực hiện vì gatePhase3. Không trảzerochannels, không inferunits, không resample. Các testchưa applicable ghiNOT_RUN/NOT_APPLICABLE chứ khôngPASS. Label-onlypriorcontrol không cần chuẩn hóa.
''')
    c.status(4,'PARTIAL',start,config,dict(numeric_adapter=True,deterministic_reread=True,physical_harmonization=False),
        ['PHASE4_HARMONIZATION_REPORT.md','adapter_configuration.json','harmonization_checks.csv','preprocessing_validation.json'],reason='Semantic harmonization gated by unknown export schema/units/task boundaries')


def lock_protocol(c,inv,decisions):
    start=time.monotonic()
    import platform
    recipe=dict(protocol_version=1,dataset='Drotar_Dobes_2020',source_commit=c.COMMIT,
        source_checksums=c.load('external_dataset_checksums.json'),dataset_inventory_sha256=c.sha(c.OUT/'DATASET_INVENTORY.json'),
        target='Workbook diag DYSGR=1,0=0; study assessment, not automatically DiaGraMo group',
        participants=120,unit='participant ID; all sessions/recordings grouped',folds=5,seed=42,
        split='StratifiedKFold5 shuffleTrue seed42',threshold='probability>0.5',
        allowed_experiments=['Train_prior_label_control'],signal_experiments_allowed=[],
        prior=dict(estimator='sklearn.dummy.DummyClassifier',strategy='prior',fit_inputs='constant zero column; labels Train only',
            normalization='none',hyperparameter_search=False,early_stopping=False,calibration_fit=False,checkpoint='JSON class-count sufficient statistics; no optimizer or RNG required for closed-form fit'),
        metrics=['accuracy','f1_macro','auc','balanced_accuracy','sensitivity','specificity','brier','log_loss','confusion','ece10'],
        uncertainty=dict(bootstrap_replicates=2000,seed=42,unit='Unstratified participant OOF rows; fixed predictions',
            cluster='Site unavailable; cannot bootstrap source clusters',scope='Conditional exploratory interval, not training/selection or domain uncertainty'),
        no_fusion=True,no_gaf_cwt_ssl=True,no_model_selection=True,source_negative_control='NOT_FEASIBLE no per-person source',
        gate_status=decisions,python=platform.python_version())
    path=c.OUT/'LOCKED_CONFIG.json'
    if path.exists():
        if c.load('LOCKED_CONFIG.json')!=recipe:raise ValueError('Locked protocol changed; refusing score-dependent mutation')
    else:c.write('LOCKED_CONFIG.json',recipe)
    digest=c.sha(path)
    predecision=dict(locked_config_sha256=digest,locked_utc=c.stamp(),
        test_model_scores_seen=False,signal_experiments='No admission: export schema/units unresolved',
        admitted='Prior-only label-balance diagnostic; does not test motor information',
        future_unlock='Requires author schema and separate new preregistered protocol; not changes based on current diagnostic results')
    if (c.OUT/'PREEXPERIMENT_DECISIONS.json').exists():
        if c.load('PREEXPERIMENT_DECISIONS.json')['locked_config_sha256']!=digest:raise ValueError('Preexperiment lock mismatch')
    else:c.write('PREEXPERIMENT_DECISIONS.json',predecision)
    c.md('LOCKED_PROTOCOL.md',f'''# Locked research protocol — before experiments

ConfigSHA256 `{digest}`; locked before any model score. Dataset120participantIDs,57positive/63negative,5folds Stratified/shuffle/seed42; repeated recordings never split bysample. Target=ExcelDYSGRvs0; not samegroundtruth as DiaGraMogroup includingongoingcases.

**Admitted:** oneTrain-prior diagnostic via DummyClassifier(strategy=prior), fitonlyTrainclasscounts, constantX, thresholdp>0.5.5fits, nohyperparametersearch/scaler/calibration/earlystop. JSONcheckpointclasscounts suffice for exact replay; optimizer/RNG resumes N/A for closed-form estimator. This is external-dataset-only internalCV labelcontrol, not Kinematic, Temporal or zero-shottransfer.

**Not admitted:** A/B/Csignalexperiments and rawKinematic/TemporalD, because tasks/exportschema/units notverified. No newencoder is built to bypass physicalmeaning. NoGAF/CWT/SSL. No50/50fusion without pairedvalidKinematic/Temporalprobabilities. No source-onlyscore absentper-personsource; never usecase/controlrecruitment route as a sourcepredictor manufactured fromlabel.

ReportAccuracy,MacroF1,AUC,BalancedAccuracy,Sensitivity,Specificity,Brier,Logloss,confusion,ECE10. OverallOOFonlyif5/5folds uniqueandcomplete. 2000participantbootstrapresamples of fixedOOF,seed42; report siteclustering unhandled and conditionalnature. No pairedcomparison with DiaGraMo258 because differentcohort/target. No clinical/generalizationclaim fromprioronly.

Budgetforecast: firstcompletedpriorfit measures throughput; before followingfits budgetadmission uses measured duration plus conservative overhead and reportingreserve. NeuralGPUtensorallocation remains0. Resumecompletedfolds fromhash-verified JSON/counts and exactprivate splitIDs; no repeatfit.
''')
    c.status(5,'COMPLETE',start,dict(locked_config_sha256=digest,admitted_experiments=recipe['allowed_experiments'],not_admitted='All signal-model comparisons'),
        dict(no_scores_used_for_selection=True),['LOCKED_PROTOCOL.md','LOCKED_CONFIG.json','PREEXPERIMENT_DECISIONS.json'])


def continue_research(c,reader,budget,load):
    inv=inventory_and_audit(c,reader,budget);print('Phase2 structural audit persisted',flush=True)
    decisions=compatibility(c,inv);print('Phase3 decisions persisted',flush=True)
    harmonize(c,reader,inv);print('Phase4 numeric adapter checked; semantic gate retained',flush=True)
    lock_protocol(c,inv,decisions);print('Phase5 immutable recipe verified',flush=True)
    experiment=load('experiments')
    if (c.OUT/'experiment_validation.json').exists() and c.load('experiment_validation.json').get('prior_control_complete'):
        # Verify existing evidence, do not refit or repeat bootstrap merely because time remains.
        import numpy as np
        import pandas as pd
        cfgsha=c.sha(c.OUT/'LOCKED_CONFIG.json')
        prior=c.load('experiment_validation.json')
        if prior['config_sha256']!=cfgsha:raise ValueError('Completed experiment signature changed')
        frame=pd.read_csv(c.LOCAL/'prior_oof.csv',dtype={'participant_id':str})
        if len(frame)!=120 or not frame.participant_id.is_unique:raise ValueError('Saved OOF incomplete')
        for fold in range(5):
            cp=c.LOCAL/f'checkpoints/prior_fold{fold}.json'
            digest=json.loads(cp.with_suffix('.sha256.json').read_text(encoding='utf-8'))['sha256']
            if c.sha(cp)!=digest:raise ValueError('Completed checkpoint changed')
            state=json.loads(cp.read_text(encoding='utf-8'))
            if state['config_sha256']!=cfgsha or state['fold']!=fold:raise ValueError('Completed checkpoint association changed')
            np.testing.assert_allclose(frame[frame.fold==fold].probability,state['probability'],rtol=0,atol=1e-15)
        print('Phase6 completed control replayed without refit/bootstrap',flush=True)
    else:experiment.run(c,budget)
    report=load('reports');report.scientific_analysis(c);print('Phase7 persisted',flush=True)
    report.final_report(c,budget);print('Phase8 persisted',flush=True)
    validate(c,budget)


def validate(c,budget):
    import subprocess
    import sys
    import psutil
    import pandas as pd
    start=time.monotonic();cpu_start=time.process_time()
    test_start=time.monotonic()
    result=subprocess.run([sys.executable,'-m','pytest','tests/external_validation','-q'],cwd=c.ROOT,capture_output=True,text=True,encoding='utf-8')
    c.target('TEST_STDOUT.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    c.write('TEST_RESULTS.json',dict(command='python -m pytest tests/external_validation -q',returncode=result.returncode,
        wall_seconds=time.monotonic()-test_start,stdout=result.stdout))
    print(result.stdout,flush=True)
    protection=c.load('PROTECTED_ARTIFACT_MANIFEST.json');changed=[]
    for item in protection['files']:
        p=c.ROOT/item['path']
        if not p.is_file() or c.sha(p)!=item['sha256']:changed.append(item['path'])
    source_errors=[]
    for item in c.load('external_dataset_checksums.json')['files']:
        if c.sha(c.LOCAL/item['name'])!=item['sha256']:source_errors.append(item['name'])
    raw=c.load('RAW_RECORDING_CHECKSUMS.json',local=True)['files']
    for item in raw:
        if c.sha(c.LOCAL/item['path'])!=item['sha256']:source_errors.append('local_raw_recording_changed')
    tracked=subprocess.check_output(['git','ls-files'],cwd=c.ROOT,text=True).splitlines()
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=c.ROOT,text=True).splitlines()
    tracked_diff=subprocess.check_output(['git','diff','--name-only'],cwd=c.ROOT,text=True).splitlines()
    external_tracked=[p for p in tracked+staged if p.startswith('data/external_datasets/')]
    private_ignored=subprocess.run(['git','check-ignore','data/external_datasets/drotar_dobes_2020/prior_oof.csv'],cwd=c.ROOT,capture_output=True).returncode==0
    leaks=[]
    # No external source user-file paths or participant clinical table schemas in public outputs.
    import re
    for p in c.OUT.rglob('*'):
        if not p.is_file() or p.suffix not in {'.json','.csv','.md','.txt'}:continue
        text=p.read_text(encoding='utf-8')
        if re.search(r'\buser\d{5}\b|\bu\d{5}s\d{5}_hw\d{5}',text):leaks.append(p.name)
        if p.suffix=='.csv' and 'participant_id' in pd.read_csv(p,nrows=0).columns:leaks.append(p.name)
    cfg=c.load('LOCKED_CONFIG.json');cfgsha=c.sha(c.OUT/'LOCKED_CONFIG.json')
    locked=cfgsha==c.load('PREEXPERIMENT_DECISIONS.json')['locked_config_sha256']==c.load('experiment_validation.json')['config_sha256']
    required=['PHASE0_ENVIRONMENT.md','HARDWARE_BUDGET.json','PROTECTED_ARTIFACT_MANIFEST.json',
        'PHASE1_DATASET_PROVENANCE.md','external_file_inventory.csv','external_schema_inventory.csv','external_label_dictionary.csv','external_dataset_checksums.json',
        'PHASE2_SIGNAL_AUDIT.md','channel_statistics.csv','timestamp_quality.csv','participant_split_integrity.json','source_label_contingency.csv',
        'PHASE3_COMPATIBILITY_REPORT.md','TASK_COMPATIBILITY_MATRIX.csv','CHANNEL_COMPATIBILITY_MATRIX.csv','TRANSFER_PROTOCOL_DECISION.json',
        'PHASE4_HARMONIZATION_REPORT.md','adapter_configuration.json','harmonization_checks.csv','preprocessing_validation.json',
        'LOCKED_PROTOCOL.md','LOCKED_CONFIG.json','PREEXPERIMENT_DECISIONS.json',
        'PHASE6_EXPERIMENT_REPORT.md','experiment_registry.csv','model_metrics.csv','fold_metrics.csv','runtime_memory.csv','checkpoint_manifest.json','experiment_validation.json',
        'PHASE7_SCIENTIFIC_ANALYSIS.md','complementarity_metrics.csv','domain_shift_summary.csv','source_confounding_summary.csv','limitations_matrix.csv',
        'EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md','EXECUTIVE_SUMMARY_VI.md','NEXT_RESEARCH_DECISION_VI.md','BLOCKERS.md','REPRODUCE.md']
    missing=[p for p in required if not (c.OUT/p).is_file()]
    checks=dict(tests_returncode=result.returncode,tests_executed=True,protected_files_checked=len(protection['files']),
        protected_files_changed=changed,source_files_changed=source_errors,source_recordings_checked=len(raw),
        locked_config_unchanged=locked,no_external_raw_tracked=not external_tracked,no_external_private_staged=not external_tracked,
        staged_files=staged,tracked_diff=tracked_diff,private_directory_git_ignored=private_ignored,public_identifier_leaks=leaks,
        missing_deliverables=missing,original_models_and_oof_unchanged=not changed,original_architecture_modified=False,
        prior_control_completed_folds=5,signal_models_fit=0,exact_frozen_transfer_complete=False,
        physical_harmonization_complete=False,signal_window_mask_tests='NOT_APPLICABLE; no semantic adapter admitted',
        signal_optimizer_resume_tests='NOT_APPLICABLE; no neural fits',source_independent_validation='UNVERIFIED',
        full_research_objective_achieved=False,technical_feasible_investigation_complete=False,
        commit_performed=False,push_or_merge_performed=False)
    checks['technical_feasible_investigation_complete']=result.returncode==0 and not changed and not source_errors and locked and not external_tracked and private_ignored and not leaks and not missing and not tracked_diff and not staged
    c.write('VALIDATION_CHECKS.json',checks)
    c.status(9,'COMPLETE' if checks['technical_feasible_investigation_complete'] else 'PARTIAL',start,
        dict(tests=result.stdout.strip(),protected_count=len(protection['files'])),checks,
        ['VALIDATION_CHECKS.json','TEST_RESULTS.json','TEST_STDOUT.txt'],reason=None if checks['technical_feasible_investigation_complete'] else 'See failed validation checks')
    stage_states={str(i):c.load(f'phase{i}_status.json')['status'] for i in range(10)}
    c.write('OVERNIGHT_RUN_STATUS.json',dict(status='COMPLETE_FEASIBLE_INVESTIGATION' if checks['technical_feasible_investigation_complete'] else 'PARTIAL',
        phases=stage_states,dataset_classification='A real raw trajectories',
        scientific_outcome='E: no currently admitted signal-learning/transfer path; prior-only control completed',
        completed_experiments=['Train_prior_label_control:5/5folds'],signal_research_complete=False,
        budget=budget.summary(),validation_wall_seconds=time.monotonic()-start,validation_cpu_seconds=time.process_time()-cpu_start,
        current_process_rss_bytes=psutil.Process().memory_info().rss,gpu_training_seconds=0,
        runtime_scope='Wall elapsed includes documentation/manual source review between CLI stages; phase timings and actual control runtime separate',
        no_further_gpu_work_scheduled=True,utc=c.stamp()))
    manifest=[]
    for directory in [c.OUT,c.ROOT/'scripts/external_validation',c.ROOT/'tests/external_validation']:
        for p in sorted(directory.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json':
                manifest.append(dict(path=p.relative_to(c.ROOT).as_posix(),bytes=p.stat().st_size,sha256=c.sha(p)))
    c.write('ARTIFACT_MANIFEST.json',dict(utc=c.stamp(),artifacts=manifest,private_artifacts_included=False))
    print(json.dumps(dict(technical_complete=checks['technical_feasible_investigation_complete'],protected_files=len(protection['files']),source_recordings=len(raw),signal_models_fit=0)),flush=True)
    if not checks['technical_feasible_investigation_complete']:raise AssertionError('Validation failed; inspect VALIDATION_CHECKS.json')
