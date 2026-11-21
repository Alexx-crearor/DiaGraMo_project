import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('phase5_test_target',ROOT/'scripts/provenance_verification/verifier.py')
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def test_join_preserves_na_prefix():
    a=pd.DataFrame({'id':['NA12345','BR12345']})
    b=pd.DataFrame({'id':['BR12345','NA12345'],'value':[2,1]})
    assert v.stable_join(a,b,'id').value.tolist()==[1,2]


def test_duplicate_join_rejected():
    with pytest.raises(ValueError):v.stable_join(pd.DataFrame({'id':['a']}),pd.DataFrame({'id':['a','a']}),'id')


def test_unmatched_join_is_visible():
    r=v.stable_join(pd.DataFrame({'id':['a']}),pd.DataFrame({'id':['b']}),'id')
    assert r['_merge'].iloc[0]=='left_only'


def test_numeric_and_missing_comparison():
    assert v.same('1',1.0) and v.same('',None)
    assert not v.same('',0) and not v.same('Diagnosis ongoing','Dysgraphia')


def test_output_path_escape():
    with pytest.raises(ValueError):v.json_out('../../must_not_write.json',{})
    with pytest.raises(ValueError):v.table('../../must_not_write.csv',[])


def test_identifier_privacy_detection():
    assert v.privacy_classification(Path('results/case.csv'),'participant_id,age\nBR12345,9')[0]=='Potentially sensitive participant-level'


def test_visual_privacy_requires_review():
    assert v.privacy_classification(Path('results/figure.png'),'')[0]=='Public-safe only after review'


def test_explicit_import_collision_guard():
    p=ROOT/'scripts/provenance_verification/run.py'
    s=importlib.util.spec_from_file_location('phase5_cli_test',p)
    cli=importlib.util.module_from_spec(s);s.loader.exec_module(cli)
    assert Path(cli.load('verifier').__file__).resolve()==p.parent/'verifier.py'
    assert Path(cli.load('reports').__file__).resolve()==p.parent/'reports.py'
    with pytest.raises(ValueError):cli.load('report')


def test_public_case_table_contains_no_identifiers():
    path=v.OUT/'DISAGREEMENT_CASE_DECISIONS.csv'
    if not path.exists():pytest.skip('Run analyze phase first')
    r=v.read_csv(path)
    assert len(r)==5 and r.case_ref.is_unique and 'participant_id' not in r
    assert r.resolution.str.startswith('UNVERIFIED').all()


def test_label_decision_does_not_overclaim():
    path=v.OUT/'LABEL_VERIFICATION_DECISION.json'
    if not path.exists():pytest.skip('Run reports phase first')
    d=json.loads(path.read_text(encoding='utf-8'))
    assert d['all_group1_confirmed_dysgraphia'] is False
    assert d['individual_case_adjudication']=='UNVERIFIED'


def test_rds_schema_or_explicit_blocker():
    path=v.OUT/'RDS_VERIFICATION_STATUS.json'
    if not path.exists():pytest.skip('Run analyze phase first')
    d=json.loads(path.read_text(encoding='utf-8'))
    assert len(d['sources'])==2
    if not d['decoded']:
        assert d['status']=='UNVERIFIED' and d['blocker']
    else:
        for source in d['sources']:
            schema=json.loads((v.PRIVATE/(Path(source['path']).stem+'_SCHEMA.json')).read_text(encoding='utf-8'))
            assert schema['objects'] and all(x['rows']>0 and x['columns'] for x in schema['objects'])


def test_evidence_status_vocabulary():
    path=v.OUT/'LABEL_EVIDENCE_MATRIX.csv'
    if not path.exists():pytest.skip('Run reports phase first')
    assert set(v.read_csv(path).verification_status)<=v.LEVELS


def test_aggregate_exception_does_not_override_identifier_guard():
    path=Path('results/cv/summary.csv')
    assert v.privacy_classification(path,'accuracy,f1_macro\n0.8,0.8')[0]=='Public-safe aggregate'
    assert v.privacy_classification(path,'participant_id\nBR12345')[0]=='Potentially sensitive participant-level'


def test_final_report_sections_and_no_encoding_damage():
    path=v.OUT/'DATASET_PROVENANCE_VERIFICATION_VI.md'
    if not path.exists():pytest.skip('Run reports phase first')
    text=path.read_text(encoding='utf-8')
    assert sum(line.startswith('## ') for line in text.splitlines())==16
    assert '\ufffd' not in text and '? ch? ??' not in text
