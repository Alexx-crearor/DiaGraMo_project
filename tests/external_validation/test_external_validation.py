import importlib.util
import io
import json
from pathlib import Path
import stat
import zipfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]


def module(name):
    spec=importlib.util.spec_from_file_location('external_test_'+name,ROOT/f'scripts/external_validation/{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


r=module('external_reader')


@pytest.mark.parametrize('name',['../escape.csv','/absolute.csv','C:/absolute.csv','safe/../../escape.csv','..\\escape.csv'])
def test_zip_traversal_rejected(tmp_path,name):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr(name,'data')
    with pytest.raises(ValueError):r.inspect_zip(p)


def test_zip_symlink_rejected(tmp_path):
    p=tmp_path/'bad.zip';i=zipfile.ZipInfo('link');i.create_system=3;i.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(p,'w') as z:z.writestr(i,'../outside')
    with pytest.raises(ValueError):r.inspect_zip(p)


@pytest.mark.parametrize('name',['a.exe','a.ps1','a.pkl'])
def test_zip_executable_rejected(tmp_path,name):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr(name,'data')
    with pytest.raises(ValueError):r.inspect_zip(p)


def test_zip_limits_and_case_collision(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('a.txt','data');z.writestr('A.txt','data')
    with pytest.raises(ValueError):r.inspect_zip(p)
    p=tmp_path/'large.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('a.csv','x'*100)
    with pytest.raises(ValueError):r.inspect_zip(p,max_uncompressed=50)


def test_safe_extraction_preserves_existing_and_refuses_overwrite(tmp_path):
    p=tmp_path/'ok.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('folder/a.csv','data')
    r.extract_safe(p,tmp_path/'out');r.extract_safe(p,tmp_path/'out')
    dest=tmp_path/'out/folder/a.csv';assert dest.read_text()=='data'
    dest.write_text('user work')
    with pytest.raises(ValueError):r.extract_safe(p,tmp_path/'out')
    assert dest.read_text()=='user work'


def test_terminal_sample_and_native_values_preserved(tmp_path):
    p=tmp_path/'sample.svc';p.write_text('3\n0 1 1000 0 12 30 0\n1 3 1008 1 12 30 8\n2 4 1016 1 12 30 9\n')
    a=r.load_svc_numeric(p)
    assert a.shape==(3,7) and a[-1,2]==1016 and a[-1,6]==9
    np.testing.assert_array_equal(a,r.load_svc_numeric(p))


@pytest.mark.parametrize('content',['2\n0 1 2 3 4 5 6\n','1\n0 1 2 3 4 5\n','1\n0 1 nan 0 4 5 6\n'])
def test_invalid_raw_rejected(tmp_path,content):
    p=tmp_path/'sample.svc';p.write_text(content)
    with pytest.raises(ValueError):r.load_svc_numeric(p)


def test_zero_sample_record_rejected_without_fabrication(tmp_path):
    p=tmp_path/'empty.svc';p.write_text('0\n')
    with pytest.raises(ValueError,match='zero declared samples'):r.load_svc_numeric(p)


def test_unknown_units_and_missing_channels_fail_closed():
    with pytest.raises(ValueError):r.require_raw_contract(['x','y','time'],True,True)
    channels=['x','y','time','pen_status','pressure','azimuth','tilt']
    with pytest.raises(ValueError):r.require_raw_contract(channels,False,True)
    with pytest.raises(ValueError):r.require_raw_contract(channels,True,False)


def test_actual_ids_labels_and_grouped_folds():
    c=module('common');p=c.LOCAL/'external_participant_inventory.csv'
    if not p.exists():pytest.skip('Run dataset audit first')
    import pandas as pd
    people=pd.read_csv(p,dtype={'participant_id':str});recordings=pd.read_csv(c.LOCAL/'external_recording_inventory.csv',dtype={'participant_id':str})
    assert len(people)==120 and people.participant_id.is_unique
    assert people.label.value_counts().to_dict()=={0:63,1:57}
    assert len(recordings)==121 and recordings.groupby('participant_id').candidate_fold.nunique().max()==1
    assert set(recordings.participant_id)==set(people.participant_id)
    for fold in range(5):assert not set(people[people.candidate_fold==fold].participant_id)&set(people[people.candidate_fold!=fold].participant_id)


def test_workbook_cached_reference_and_unknown_label():
    c=module('common')
    sheets=r.xlsx_cells(c.LOCAL/'data2_SciRep_pub.xlsx')
    people=r.workbook_participants(sheets)
    assert len(people)==120 and people.label_formula_present.sum()==1
    assert people.id_padding_applied.sum()==49 and people.participant_id.str.len().eq(5).all()
    # Reject formulas whose cached value cannot be verified by a simple source reference.
    for cells in sheets.values():
        next(x for x in cells if x['formula_present'])['formula']='SOME_EXTERNAL_FUNCTION()'
    with pytest.raises(ValueError):r.workbook_participants(sheets)


def test_probabilities_single_class_and_invalid_inputs():
    m=module('experiments')
    assert m.metrics([0,1],[.1,.9])['accuracy']==1
    assert m.metrics([1,1],[.9,.8])['auc'] is None
    with pytest.raises(ValueError):m.metrics([0,1],[.1,float('nan')])


def test_lock_preserves_no_signal_admission():
    c=module('common');p=c.OUT/'LOCKED_CONFIG.json'
    if not p.exists():pytest.skip('Run lock phase first')
    cfg=c.load('LOCKED_CONFIG.json')
    assert cfg['signal_experiments_allowed']==[] and cfg['allowed_experiments']==['Train_prior_label_control']
    assert c.sha(p)==c.load('PREEXPERIMENT_DECISIONS.json')['locked_config_sha256']


def test_explicit_import_and_output_path_guards():
    cli=module('run');c=cli.load('common')
    assert Path(cli.load('external_reader').__file__).parent==ROOT/'scripts/external_validation'
    with pytest.raises(ValueError):cli.load('report')
    with pytest.raises(ValueError):c.target('../../escape.json')


def test_no_incomplete_results_claimed_complete():
    c=module('common');p=c.OUT/'experiment_validation.json'
    if not p.exists():pytest.skip('Run admitted experiment first')
    v=c.load('experiment_validation.json')
    assert v['prior_control_complete'] and v['completed_folds']==5 and v['participants']==120
    assert v['signal_models_fit']==0 and v['full_requested_signal_comparison_complete'] is False


def test_runtime_budget_fails_closed():
    c=module('common')
    with pytest.raises(ValueError):c.Budget(0)
    b=c.Budget(8,15)
    with pytest.raises(TimeoutError):b.admit(8*3600)


def test_prior_oof_matches_train_counts_and_checkpoint_hashes():
    c=module('common');p=c.LOCAL/'prior_oof.csv'
    if not p.exists():pytest.skip('Run admitted experiment first')
    import pandas as pd
    people=pd.read_csv(c.LOCAL/'external_participant_inventory.csv',dtype={'participant_id':str})
    oof=pd.read_csv(p,dtype={'participant_id':str})
    assert len(oof)==120 and oof.participant_id.is_unique
    for fold in range(5):
        expected=people[people.candidate_fold!=fold].label.mean()
        np.testing.assert_allclose(oof[oof.fold==fold].probability,expected,rtol=0,atol=1e-15)
        cp=c.LOCAL/f'checkpoints/prior_fold{fold}.json'
        meta=json.loads(cp.with_suffix('.sha256.json').read_text(encoding='utf-8'))
        assert c.sha(cp)==meta['sha256']


def test_final_report_has_twenty_sections_and_no_raw_id_paths():
    c=module('common');p=c.OUT/'EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md'
    if not p.exists():pytest.skip('Generate report first')
    text=p.read_text(encoding='utf-8')
    assert sum(line.startswith('## ') for line in text.splitlines())==20
    assert '\ufffd' not in text and 'user00135' not in text
