"""Regression checks for the non-destructive cleanup and imported CLI."""
from pathlib import Path
import importlib.util
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def test_imported_cli_help_never_reads_signals(monkeypatch,capsys):
    from scripts.native_channel_research import run
    def forbidden():raise AssertionError('Help must not read or train')
    monkeypatch.setattr(run,'load_data',forbidden)
    with pytest.raises(SystemExit) as e:run.main(['--help'])
    assert e.value.code==0 and '--max-runtime-hours' in capsys.readouterr().out


def test_imported_cli_invalid_flag_never_reads_signals(monkeypatch):
    from scripts.native_channel_research import run
    monkeypatch.setattr(run,'load_data',lambda:pytest.fail('Unexpected data access'))
    with pytest.raises(SystemExit) as e:run.main(['--not-an-existing-option'])
    assert e.value.code==2


def test_research_registry_has_scope_and_authoritative_sources():
    import pandas as pd
    f=pd.read_csv(ROOT/'docs/research/EXPERIMENT_REGISTRY.csv')
    assert len(f)>=168 and {'source_commit','metrics_source','evaluation_scope','status'}<=set(f)
    for path in f.metrics_source.unique():assert (ROOT/path).is_file()
    for path in f.report_path.unique():assert (ROOT/path).is_file()
    assert set(['Kinematic_LightGBM','Stroke_Temporal','Fixed50_Fusion','Slovak_B3_OrderedCNN'])<=set(f.model)
    assert 'INCOMPLETE_PILOT_ONLY' in set(f.status)


def test_new_markdown_links_are_local_and_resolve():
    from scripts.research_cleanup.validate import links
    result=links();assert not result['broken'],result


def test_read_only_oof_replay():
    from scripts.research_cleanup.validate import numerical
    result=numerical()
    assert result['B3_B4_binary_identical'] and result['fusion_delta_ci_crosses_zero']
    assert result['new_model_fit'] is False and result['bootstrap_recomputed'] is False


def test_documentation_does_not_relabel_prior_xai_as_new():
    report=(ROOT/'docs/weekly_reports/2026-11-16_to_2026-11-22/WEEKLY_PROGRESS_VI.md').read_text(encoding='utf8')
    assert '80f9493' in report and 'mốc' in report
    assert 'task-level Transformer' in report and 'không' in report
    assert 'INCOMPLETE' in report and 'fixed' in report.lower()


def test_private_path_classification_and_safe_aliases():
    from scripts.research_cleanup.audit import private,safe_name
    assert private('data/example.csv')
    assert private('results/example/XX12345.png')
    assert 'XX12345' not in safe_name('results/example/XX12345.png')
    assert not private('src/diagramo/features.py')


def test_before_snapshot_is_immutable_and_tracks_original_head():
    from scripts.research_cleanup.audit import digest,LOCAL
    f=json.loads((ROOT/'docs/research_cleanup/PROTECTED_BEFORE.json').read_text(encoding='utf8'))
    assert f['head']=='abee084d6621ccf4175f3e4a1a5a484756bb824f'
    assert digest(LOCAL/'protected_exact.json')==f['exact_manifest_sha256']
