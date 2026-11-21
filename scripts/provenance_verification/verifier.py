"""Phase 5: bounded read-only provenance checks; no model imports or network data fetches."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/provenance_verification'
PRIVATE = ROOT / 'data/provenance_verification_local'
V3 = ROOT / 'results/data_audit_v3'
LEVELS = {'VERIFIED', 'SUPPORTED', 'UNVERIFIED', 'CONTRADICTORY'}
COMPONENTS = ['group_problems_flag_to_omit', 'non_czech_language_flag_to_omit',
              'special_care_type_not_relevant_flag_to_omit', 'care_type_missing_flag_to_omit']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_out(name, value, private=False):
    base = PRIVATE if private else OUT
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Output path escaped its designated directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(name, rows, private=False):
    base = PRIVATE if private else OUT
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Unsafe CSV output')
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False)


def stable_join(left, right, key):
    """Never choose an arbitrary row from a duplicated source."""
    if not left[key].is_unique or not right[key].is_unique:
        raise ValueError('Ambiguous source join; investigate duplicates explicitly')
    return left.merge(right, on=key, how='left', validate='one_to_one', indicator=True)


def same(a, b):
    if a is None or pd.isna(a) or str(a).strip() in ('', 'NA'):
        return b is None or pd.isna(b) or str(b).strip() in ('', 'NA')
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def meta(pid):
    d = json.loads((ROOT / f'data/clean_data/{pid}_merged.json').read_text(encoding='utf-8'))
    values = [v for k, v in d.items() if k.endswith('_cognitive_test_data') and v]
    if len(values) != 1 or len(values[0]) != 1:
        raise ValueError('Cognitive metadata selection is ambiguous')
    return values[0][0]


def initialize():
    if (PRIVATE / 'PROTECTED_BEFORE.json').exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    names = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    names += [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'data').glob('*/cognitive_test_data/*') if p.is_file()]
    names += ['data/derived/participants.csv', 'data/derived/schema.json', 'data/derived/kinematic.npz']
    ids = set(read_csv(V3 / 'label_disagreements.csv').participant_id)
    ids.update(read_csv(V3 / 'duplicate_metadata_ids.csv').participant_id)
    names += [f'data/clean_data/{pid}_merged.json' for pid in sorted(ids)]
    files = []
    for name in sorted(set(names)):
        if name.startswith(('results/provenance_verification/', 'scripts/provenance_verification/', 'tests/provenance_verification/')):
            continue
        p = ROOT / name
        if not p.is_file():
            raise FileNotFoundError(name)
        files.append(dict(path=name, sha256=sha(p), bytes=p.stat().st_size))
    json_out('PROTECTED_BEFORE.json', files, private=True)
    json_out('EXECUTION_CONTEXT.json', dict(reference_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        initial_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
        protected_files=len(files), protected_bytes=sum(x['bytes'] for x in files),
        utc=datetime.now(timezone.utc).isoformat(), models_fit=False, gpu_used=False,
        source_data_downloaded=False, private_output='data/provenance_verification_local/ (Git ignored)'))


def inspect_local():
    raw = pd.read_csv(ROOT / 'data/raw_data/cognitive_test_data/cognitiveTestCSV2data.csv', sep=';', keep_default_na=False)
    clean = pd.read_csv(ROOT / 'data/processed_data/cognitive_test_data/cognitiveTestCSV2data_cleaned.csv', sep=';', keep_default_na=False)
    folds = read_csv(ROOT / 'results/cv/fold_assignments.csv')
    cohort = read_csv(V3 / 'audit_cohort.csv')
    joined = stable_join(folds, cohort[['participant_id', 'group']], 'participant_id')
    assert joined['_merge'].eq('both').all() and len(joined) == 258
    assert all(same(a, b) for a, b in zip(joined.label, joined.group))
    disagreements = read_csv(V3 / 'label_disagreements.csv')
    local_rows, public_rows = [], []
    for number, pid in enumerate(sorted(disagreements.participant_id), 1):
        m = meta(pid)
        r = raw[raw.id_participant.eq(pid)]
        c = clean[clean.id_participant.eq(pid)]
        if len(r) != 1 or len(c) != 1:
            raise ValueError('Disagreement case not uniquely linked')
        fields = ['group', 'special_care_type', 'special_care_type_dysgraphia',
                  'special_care_type_diagnosis_ongoing', 'special_care_type_dyslexia',
                  'special_care_type_dysortographia', 'special_care_type_dyscalculia',
                  'special_care_type_dyspraxia', 'special_care_type_typical_development']
        matches = all(same(r.iloc[0][k], m.get(k)) and same(c.iloc[0][k], m.get(k)) for k in fields)
        ongoing = same(m.get('special_care_type_diagnosis_ongoing'), 1)
        interpretation = 'SUPPORTED: compatible with published ongoing-diagnosis inclusion' if ongoing else 'CONTRADICTORY: group/specific-care fields require owner adjudication'
        case_ref = f'PV-CASE-{number:02d}'
        local_rows.append(dict(case_ref=case_ref, participant_id=pid, **{k:m.get(k) for k in fields},
            **{k:m.get(k) for k in COMPONENTS}, raw_unique_rows=len(r), cleaned_unique_rows=len(c),
            three_source_fields_match=matches, interpretation=interpretation,
            adjudication='UNRESOLVED; no labels changed', missing_fields='|'.join(k for k in fields if m.get(k) is None)))
        # Public table deliberately excludes actual IDs, source/demographic/clinical rows and direction of discordance.
        public_rows.append(dict(case_ref=case_ref, join_status='VERIFIED' if matches else 'CONTRADICTORY',
            interpretation=interpretation, resolution='UNVERIFIED: individual clinical record not available',
            action='Keep protected label; request owner adjudication'))
    table('UNRESOLVED_LABEL_CASES.csv', local_rows, private=True)
    table('DISAGREEMENT_CASE_DECISIONS.csv', public_rows)
    duplicate_rows = []
    for pid in raw.loc[raw.id_participant.duplicated(keep=False), 'id_participant'].unique():
        r = raw[raw.id_participant.eq(pid)]
        c = clean[clean.id_participant.eq(pid)]
        m = meta(pid)
        differing = [k for k in raw if r[k].astype(str).nunique() > 1]
        matching = [i+1 for i, (_, row) in enumerate(r.iterrows()) if all(same(row[k],m.get(k)) for k in differing)]
        duplicate_rows.append(dict(participant_id=pid, raw_rows=len(r), cleaned_rows=len(c),
            conflicting_fields='|'.join(differing), merged_matches_raw_occurrences='|'.join(map(str,matching)),
            protected_cohort_member=pid in set(folds.participant_id),
            relevant_label_conflicts=any(k in differing for k in ['group','special_care_type_dysgraphia'])))
    table('DUPLICATE_METADATA_LOCAL.csv', duplicate_rows, private=True)
    # Aggregate checks across all cleaned rows, then across the protected cohort. Empirical equivalence is not a rule proof.
    assert set(COMPONENTS + ['flag_to_omit_total']).issubset(clean.columns)
    observed_or = clean[COMPONENTS].apply(pd.to_numeric, errors='coerce').eq(1).any(axis=1)
    total = pd.to_numeric(clean.flag_to_omit_total, errors='coerce')
    complete = clean[COMPONENTS].apply(pd.to_numeric, errors='coerce').notna().all(axis=1) & total.notna()
    omission = dict(rows=len(clean), columns=len(clean.columns), raw_rows=len(raw), raw_columns=len(raw.columns),
        raw_has_total='flag_to_omit_total' in raw, cleaned_has_total=True,
        component_or_disagreements=int((observed_or[complete] != total[complete].eq(1)).sum()),
        complete_component_rows=int(complete.sum()), incomplete_component_rows=int((~complete).sum()),
        flagged_rows=int(total.eq(1).sum()), flagged_unique_ids=int(clean.loc[total.eq(1),'id_participant'].nunique()),
        protected_flagged_unique_ids=int(clean.loc[total.eq(1)&clean.id_participant.isin(folds.participant_id),'id_participant'].nunique()),
        formula_status='SUPPORTED empirical OR agreement only; original R source not inspected',
        raw_unique_ids=int(raw.id_participant.nunique()), cleaned_unique_ids=int(clean.id_participant.nunique()),
        duplicate_count=len(duplicate_rows), duplicate_summary=[{k:v for k,v in x.items() if k!='participant_id'} for x in duplicate_rows],
        disagreement_cases=len(local_rows), ongoing_cases=sum('ongoing-diagnosis' in x['interpretation'] for x in local_rows),
        all_disagreement_sources_match=all(x['three_source_fields_match'] for x in local_rows),
        folds_and_labels_match_v3=True)
    # Check cleaned flags against every merged row from the 258 cohort, without rewriting input.
    checks = []
    prior=read_csv(V3/'label_consistency.csv').set_index('participant_id')
    for pid in folds.participant_id:
        rows=clean[clean.id_participant.eq(pid)]
        m=prior.loc[pid].to_dict()
        m['special_care_type_dysgraphia']=m['diagnosis']
        keys=COMPONENTS+['flag_to_omit_total','group','special_care_type_dysgraphia']
        checks.append(dict(unique_rows=len(rows)==1,
            all_fields_match=len(rows)>0 and all(any(same(v,m.get(k)) for v in rows[k]) for k in keys)))
    omission['cleaned_to_merged_cohort_all_fields_match']=all(x['all_fields_match'] for x in checks)
    omission['cleaned_to_merged_nonunique_ids']=sum(not x['unique_rows'] for x in checks)
    omission['cleaned_to_merged_comparison_scope']='Protected V3 metadata snapshot for 258; direct merged JSON read for five disagreements and duplicate only'
    json_out('LOCAL_SOURCE_CHECKS.json', omission)
    return omission


def rds_status():
    availability = {k:importlib.util.find_spec(k) is not None for k in ['pyreadr','rdata']}
    result=dict(readers=availability, rscript=shutil.which('Rscript'), decoded=False,
                installed_dependencies=False, sources=[], status='UNVERIFIED')
    for p in sorted((ROOT/'data').glob('*/cognitive_test_data/*.rds')):
        entry=dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p), decoded=False)
        if availability['pyreadr']:
            import pyreadr
            try:
                objects=pyreadr.read_r(str(p))
                entry['objects']=[dict(name=str(k), type=type(v).__name__, rows=len(v), columns=list(v.columns),
                    dtypes={str(k):str(t) for k,t in v.dtypes.items()},
                    factor_levels={str(k):list(map(str,v[k].cat.categories)) for k in v.select_dtypes('category')}) for k,v in objects.items()]
                entry['decoded']=True
                json_out(p.stem+'_SCHEMA.json', entry, private=True)
            except Exception as e:
                entry['reader_error']=type(e).__name__
        result['sources'].append({k:v for k,v in entry.items() if k!='objects'})
    result['decoded']=bool(result['sources']) and all(x['decoded'] for x in result['sources'])
    result['status']='VERIFIED' if result['decoded'] else 'UNVERIFIED'
    result['blocker']=None if result['decoded'] else 'No working RDS decode performed. pyreadr/rdata absent and Rscript unavailable in inspected environment; no install authorized. CSV is not an RDS schema substitute.'
    json_out('RDS_VERIFICATION_STATUS.json', result)
    return result


def privacy_classification(path, content):
    low=content.lower()
    identifiers=bool(re.search(r'\b(?:BR|HK|NA|PR|OV|OL|ZL|CB|PL|PA|JI|LI|KV|UH|TR)\d{4,}\b',content))
    row_schema=any(k in low for k in ['participant_id','id_participant','subject_id'])
    demographics=any(k in low for k in ['sex_female','id_school','special_care_type','date_of_administration','age,'])
    if identifiers or (row_schema and demographics):
        return 'Potentially sensitive participant-level', 'Linkable pseudonyms/individual attributes; local-release equivalence pending'
    reviewed_aggregates={
        'results/cv/summary.csv',
        'results/data_audit_v3/source_association_metrics.csv',
        'results/data_audit_v3/metadata_shortcut_metrics.csv',
        'results/data_audit_v3/matched_model_metrics.csv',
    }
    if path.as_posix() in reviewed_aggregates:
        return 'Public-safe aggregate', 'Reviewed cohort-level metrics only; no individual rows or small source-specific cells'
    if path.suffix.lower() in ('.png','.svg','.ipynb','.npz','.npy','.pt','.joblib','.pkl','.xlsx'):
        return 'Public-safe only after review', 'Binary/visual/embedded content requires manual review; weights are not automatically anonymous'
    if path.suffix.lower() in ('.csv','.json') and row_schema:
        return 'Must remain local pending permission', 'Participant schema present; actual linkage/consent must be reviewed'
    if path.suffix.lower() in ('.csv','.md','.json','.txt') and path.parts[0]=='results':
        return 'Public-safe only after review', 'Aggregate candidate; automated screening cannot certify privacy'
    return 'License status unknown', 'Code/document license and provenance must be checked separately'


def privacy_inventory():
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    rows=[]
    for name in names:
        p=ROOT/name
        if not p.is_file():continue
        if not name.startswith(('results/','notebooks/','docs/','references/')):continue
        content=p.read_text(encoding='utf-8',errors='replace') if p.suffix.lower() in ('.csv','.json','.md','.txt','.ipynb','.svg','.log') else ''
        category,reason=privacy_classification(Path(name),content)
        rows.append(dict(path=name,classification=category,reason=reason,
            scope='Tracked current tree; no raw row reproduced',license='Dataset CC BY 4.0 verified; local version/derived release scope UNVERIFIED',
            recommendation='Review existing Git history before any publication; no deletion/history rewrite performed'))
    table('ARTIFACT_PRIVACY_CLASSIFICATION_EXACT_LOCAL.csv',rows,private=True)
    ids=sorted(set(read_csv(V3/'participants_inventory.csv').participant_id),key=len,reverse=True)
    for row in rows:
        for pid in ids:
            row['path']=row['path'].replace(pid,'[participant-id]')
    table('ARTIFACT_PRIVACY_CLASSIFICATION.csv',rows)
    return rows


def validate():
    before=json.loads((PRIVATE/'PROTECTED_BEFORE.json').read_text(encoding='utf-8'))
    changes=[x['path'] for x in before if not (ROOT/x['path']).is_file() or sha(ROOT/x['path'])!=x['sha256']]
    local=json.loads((OUT/'LOCAL_SOURCE_CHECKS.json').read_text(encoding='utf-8'))
    rds=json.loads((OUT/'RDS_VERIFICATION_STATUS.json').read_text(encoding='utf-8'))
    checks=dict(protected_files_checked=len(before), protected_changed=changes, models_fit=False,
        old_audit_rerun=False, clinical_provenance_fully_verified=False, source_causation_verified=False,
        rds_decoded=rds['decoded'], local_join_checks_passed=local['all_disagreement_sources_match'] and local['folds_and_labels_match_v3'],
        individual_clinical_adjudication_complete=False, no_commit_or_push=True,
        local_only_git_ignored=subprocess.run(['git','check-ignore','data/provenance_verification_local/UNRESOLVED_LABEL_CASES.csv'],cwd=ROOT,capture_output=True).returncode==0)
    # Reject any public artifact leaking actual protected participant IDs (including paths to source individuals).
    ids=set(read_csv(V3/'participants_inventory.csv').participant_id)
    leaks=[]
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.md','.json','.txt') and p.name not in ('ARTIFACT_MANIFEST.json','VALIDATION_CHECKS.json'):
            content=p.read_text(encoding='utf-8')
            if any(re.search(r'(?<![A-Za-z0-9])'+re.escape(pid)+r'(?![A-Za-z0-9])',content) for pid in ids):leaks.append(p.name)
    checks['public_artifact_identifier_leaks']=leaks
    checkdiff=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    checks['tracked_diff']=checkdiff
    required=['ORIGINAL_DATASET_RESEARCH_VI.md','SOURCE_DOCUMENTS.csv','ACQUISITION_PROVENANCE_MATRIX.csv',
        'LABEL_PROVENANCE_VI.md','LABEL_EVIDENCE_MATRIX.csv','LABEL_VERIFICATION_DECISION.json',
        'RDS_SOURCE_TRACE_VI.md','OMISSION_RULE_PROVENANCE.csv','DUPLICATE_METADATA_INVESTIGATION.md',
        'SOURCE_CONFOUNDING_INTERPRETATION_VI.md','HYPOTHESIS_EVIDENCE_MATRIX.csv','IDENTIFIABILITY_DECISION.md',
        'PUBLIC_DATA_RELEASE_AUDIT_VI.md','ARTIFACT_PRIVACY_CLASSIFICATION.csv','REPOSITORY_REMEDIATION_PLAN.md',
        'EXTERNAL_VALIDATION_REQUIREMENTS.md','DATASET_COMPATIBILITY_MATRIX.csv','EXTERNAL_DATASET_SHORTLIST.md',
        'VALIDATION_READINESS_DECISION.md','DATASET_PROVENANCE_VERIFICATION_VI.md','AUTHOR_QUESTIONS_VI.md',
        'NEXT_RESEARCH_DECISION_VI.md','BLOCKERS.md','REPRODUCE.md']
    checks['missing_deliverables']=[x for x in required if not (OUT/x).is_file()]
    test=json.loads((OUT/'TEST_RESULTS.json').read_text(encoding='utf-8')) if (OUT/'TEST_RESULTS.json').exists() else {'returncode':None}
    checks['tests']=test
    checks['technical_complete']=not changes and not leaks and not checks['missing_deliverables'] and not checkdiff and checks['local_join_checks_passed'] and checks['local_only_git_ignored'] and test['returncode']==0
    checks['scientific_decision']='CONDITIONAL GO methodological; NO-GO confirmed-diagnosis or source-invariant clinical claims'
    json_out('VALIDATION_CHECKS.json',checks)
    manifest=[]
    for folder in [OUT,ROOT/'scripts/provenance_verification',ROOT/'tests/provenance_verification']:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json':
                manifest.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    json_out('ARTIFACT_MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(), artifacts=manifest,
        private_sources_manifest='Kept local in ignored data directory; public manifest contains no source participant paths'))
    if not checks['technical_complete']:raise AssertionError(checks)
    return checks
