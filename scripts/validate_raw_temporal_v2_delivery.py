"""Final delivery gate. Never turn partial experiments into a completed study."""
import argparse
import subprocess
import xml.etree.ElementTree as ET
import nbformat
from raw_temporal_v2_common import *


NOTEBOOK = ROOT/'notebooks/07_raw_temporal_research_recovery.ipynb'


def notebook_check():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert cells and all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == 'error' for cell in cells for output in cell.outputs)
    assert any(output.output_type in ('display_data', 'execute_result') and
               'text/html' in output.get('data', {}) for cell in cells for output in cell.outputs)
    source = '\n'.join(cell.source for cell in cells)
    for name in ['phase4_nested_comparison.csv', 'phase4_paired_deltas.csv',
                 'phase4_fold_results.csv', 'FINAL_DECISION_VI.md']:
        assert name in source, name
    # Status must be materialized in saved outputs, not merely source code.
    status_cells = [cell for cell in cells if "OUT / 'validation.json'" in cell.source]
    assert len(status_cells) == 1 and status_cells[0].outputs
    return dict(path=str(NOTEBOOK.relative_to(ROOT)), sha256=sha(NOTEBOOK),
                executed_code_cells=len(cells), error_outputs=0)


def preserved_files():
    expected = json.loads((OUT/'protected_git_tree.json').read_text())
    allowed = {'.gitignore', '.gitattributes'}
    for path, digest in expected.items():
        if path in allowed: continue
        current = subprocess.check_output(['git', 'hash-object', '--path='+path, path], cwd=ROOT, text=True).strip()
        assert current == digest, ('Original tracked file changed', path)
    manifests = [json.loads((OLD/'protected_manifest.json').read_text())]
    original = json.loads((OLD/'reproducibility_manifest.json').read_text())
    manifests += [original['source_sha256'], original['artifact_sha256']]
    sources = json.loads((OLD/'audit/source_hashes.json').read_text())
    manifests += [{row['path']: row['sha256'] for row in sources}]
    checked = {}
    for manifest in manifests:
        for path, digest in manifest.items():
            assert not (path in checked and checked[path] != digest), path
            if path in checked: continue
            assert sha(ROOT/path) == digest, ('Original artifact changed', path)
            checked[path] = digest
    return dict(original_tracked_files=len(expected)-len(allowed),
                original_source_and_artifact_files=len(checked), all_unchanged=True,
                permitted_root_metadata_changes=sorted(allowed))


def main(refresh_only=False):
    destination = OUT/'delivery_validation.json'
    if refresh_only:
        prior = json.loads(destination.read_text()); assert prior['complete']
        assert json.loads((OUT/'validation.json').read_text())['complete']
        prior['notebook'] = notebook_check()
        prior['notebook_refreshed_after_completion'] = True
        write(destination, prior)
        return
    for phase in range(5):
        assert json.loads((OUT/f'phase{phase}_validation.json').read_text())['complete'], phase
    from raw_temporal_v2_evaluation import CONFIG_PATH, SOURCES
    model = json.loads((OUT/'phase4_complementarity/model_validation.json').read_text())
    assert model['complete'] and model['source_signature'] == signature(CONFIG_PATH, SOURCES)['signature']
    for path, digest in model['verified_artifacts'].items():
        assert sha(ROOT/path) == digest, path
    xai = json.loads((OUT/'phase4_complementarity/xai/validation.json').read_text())
    assert xai['complete'] and not xai.get('figures_pending', True)
    active = json.loads((OUT/'phase4_complementarity/xai/active_run.json').read_text())
    run = ROOT/active['run']
    assert sha(run/'validation.json') == xai['run_validation_sha256']
    xai_run = json.loads((run/'validation.json').read_text())
    for path, digest in xai_run['verified_artifacts'].items():
        assert sha(ROOT/path) == digest, path
    assert sha(run/'figure_manifest.json') == xai['figure_manifest_sha256']
    for figure in json.loads((run/'figure_manifest.json').read_text()):
        for path, digest in figure['sources'].items(): assert sha(ROOT/path) == digest, path
        for output in figure['outputs'].values(): assert sha(ROOT/output['path']) == output['sha256']
    external = json.loads((OUT/'phase4_complementarity/external/compatibility_audit.json').read_text())
    assert external['status'] == 'BLOCKED_DATA_NOT_AVAILABLE' and not external['external_metrics_created']
    required = [*(f'PHASE{i}_REPORT_VI.md' for i in range(5)), 'REPORT_VI.md',
                'FINAL_DECISION_VI.md', 'final_decision.json', 'environment_versions.json',
                'tables/phase4_nested_comparison.csv', 'tables/phase4_fold_results.csv',
                'tables/phase4_paired_deltas.csv', 'tables/phase4_fusion_weights.csv']
    for path in required:
        assert (OUT/path).stat().st_size > 0, path
    ledger = pd.read_csv(OUT/'EXPERIMENT_LEDGER.csv').fillna('')
    unfinished = ledger[ledger.training_status.isin(['planned', 'running', 'failed', 'cpu_running', 'cpu_failed'])]
    assert unfinished.empty, unfinished[['experiment_id', 'training_status']].to_dict('records')
    figures = list((OUT/'figures').glob('*.png')) + list((run/'figures').glob('*.png'))
    assert figures
    from PIL import Image
    for path in figures:
        with Image.open(path) as picture:
            picture.verify()
        svg = path.with_suffix('.svg'); assert svg.exists()
        ET.parse(svg)
    tests = sorted((ROOT/'tests').glob('test_raw_temporal_v2*.py'))
    run = subprocess.run([sys.executable, '-m', 'pytest', *map(str, tests), '-q'], cwd=ROOT,
                         capture_output=True, text=True)
    (OUT/'delivery_tests.log').write_text(run.stdout+'\n'+run.stderr, encoding='utf-8')
    assert run.returncode == 0, 'See delivery_tests.log'
    preservation = preserved_files()
    notebook = notebook_check()
    phase4 = json.loads((OUT/'phase4_validation.json').read_text())
    phase4['delivery_notebook_validation_pending'] = False
    write(OUT/'phase4_validation.json', phase4)
    write(destination, dict(complete=True, preservation=preservation, notebook=notebook,
        tests_passed=True, test_files=len(tests), figure_pairs=len(figures),
        model_artifacts_rechecked=len(model['verified_artifacts']),
        external_validation_status=external['status'], inference_scope='Exploratory internal CV',
        failed_attribution_diagnostics_not_interpreted_as_valid=True,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        remote_push_performed=False, notebook_refreshed_after_completion=False))
    status = json.loads((OUT/'validation.json').read_text())
    status.update(complete=True, unfinished=[], delivery_validation='complete',
                  external_evaluation_status=external['status'])
    write(OUT/'validation.json', status)
    print('DELIVERY VALIDATED; external evaluation explicitly blocked, internal conclusions exploratory', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--notebook-refresh-only', action='store_true')
    args = parser.parse_args()
    with gpu_worker(): main(args.notebook_refresh_only)
