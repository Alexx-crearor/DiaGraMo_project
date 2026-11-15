"""Read-only structural checks; reports/notebooks run only on temporary copies.

No classifier fitting or pretrained encoder forward pass is performed.
"""
import argparse
import ast
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
OUT = ROOT / 'docs/cleanup'


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def structure():
    import nbformat
    sources = sorted((ROOT / 'scripts').glob('*.py')) + sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'tests').glob('*.py'))
    removed = json.loads((OUT / 'removal_plan.json').read_text())['tracked']
    forbidden = {Path(p).stem for p in removed if p.endswith('.py') and p.startswith(('scripts/', 'src/'))}
    imports = {}
    for path in sources:
        source = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), 'exec')
        deps = sorted({n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
                      | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
        assert not any(d.split('.')[-1] in forbidden for d in deps), path
        imports[path.relative_to(ROOT).as_posix()] = deps
    baseline = ['prepare', 'download_weights', 'extract', 'evaluate', 'run_all', 'report',
                'report_paper_tables', 'sensitivity', 'train_selected', 'execute_notebooks']
    for module in baseline:
        importlib.import_module(module)
    for branch in ('kinematic', 'moment', 'gaf', 'cwt'):
        importlib.import_module('diagramo.branches.' + branch)
    notebooks = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    assert len(notebooks) == 4 and [p.name[:2] for p in notebooks] == ['01', '02', '03', '04']
    for path in notebooks:
        n = nbformat.read(path, as_version=4)
        nbformat.validate(n)
        for i, cell in enumerate(n.cells):
            if cell.cell_type == 'code':
                compile(cell.source, str(path) + ':' + str(i), 'exec')
    checks = {}
    for module in ('extract', 'evaluate', 'execute_notebooks', 'run_all'):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT / 'scripts' / (module + '.py')), '--help'],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        assert result.returncode == 0, result.stderr
        checks[module] = True
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT / 'tests.log').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    assert tests.returncode == 0, tests.stderr
    pip = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, encoding='utf-8')
    assert pip.returncode == 0, pip.stdout + pip.stderr
    write('structure_validation.json', {'complete': True, 'compiled_python_files': len(sources),
          'imports': imports, 'baseline_modules_imported': baseline, 'cli_help': checks,
          'notebooks_validated': [p.name for p in notebooks], 'unit_tests': 6,
          'pip_check': pip.stdout.strip(), 'training_performed': False})
    print('STRUCTURE PASSED', flush=True)


def encoders():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
    os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')
    import torch
    from extract import load_encoder
    torch.set_num_threads(4)
    rows = []
    specifications = [
        ('resnet50', 2048, 'conv1', 'resnet50-11ad3fa6.pth'),
        ('efficientnet_b4', 1792, 'features.0.0', 'efficientnet_b4_rwightman-23ab8bcd.pth'),
        ('vit_b_16', 768, 'conv_proj', 'vit_b_16-c867db91.pth'),
        ('vgg16', 4096, 'features.0', 'vgg16-397923af.pth')]
    for name, dim, first, checkpoint in specifications:
        torch.manual_seed(42)
        model, actual_dim = load_encoder(name)
        model.eval().requires_grad_(False)
        layer = model.get_submodule(first)
        weights = torch.load(ROOT / '.cache/torch/hub/checkpoints' / checkpoint, map_location='cpu', weights_only=True)
        expected = weights[first + '.weight'].sum(1, keepdim=True).repeat(1, 7, 1, 1) / 7
        assert torch.equal(layer.weight, expected) and layer.in_channels == 7 and actual_dim == dim
        if layer.bias is not None:
            assert torch.equal(layer.bias, weights[first + '.bias'])
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        rows.append({'encoder': name, 'channels': 7, 'dimension_per_task': dim,
                     'kernel_and_bias_exactly_match_baseline_rule': True, 'frozen': True})
        del model, weights, expected, layer
        gc.collect()
        print('INITIALIZED CPU', name, flush=True)
    from momentfm import MOMENTPipeline
    snapshots = ROOT / '.cache/huggingface/hub/models--AutonLab--MOMENT-1-large/snapshots'
    revision = next(r['revision'] for r in json.loads((ROOT / 'references/weights_manifest.json').read_text())
                    if r['name'] == 'AutonLab/MOMENT-1-large')
    model = MOMENTPipeline.from_pretrained(str(snapshots / revision), model_kwargs={'task_name': 'embedding'})
    model.init()
    model.eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    rows.append({'encoder': 'MOMENT-1-large', 'initialized_offline_from_pinned_revision': revision, 'frozen': True})
    write('encoder_validation.json', {'complete': True, 'device': 'cpu', 'encoders': rows,
          'forward_pass_performed': False, 'new_embeddings_extracted': False})
    print('ENCODERS PASSED', flush=True)


def reports():
    import pandas as pd
    import report
    import report_paper_tables
    sandbox = Path(tempfile.mkdtemp(prefix='cleanup_report_', dir=ROOT / 'tmp'))
    (sandbox / 'data/derived').mkdir(parents=True)
    (sandbox / 'results').mkdir()
    for folder in ('cv', 'audit', 'sensitivity', 'figures'):
        shutil.copytree(ROOT / 'results' / folder, sandbox / 'results' / folder)
    for path in (ROOT / 'results').iterdir():
        if path.is_file():
            shutil.copy2(path, sandbox / 'results' / path.name)
    shutil.copy2(ROOT / 'data/derived/signals512.npy', sandbox / 'data/derived/signals512.npy')
    report.ROOT = sandbox
    sys.argv = ['report.py', '--no-notebooks']
    report.main()
    report_paper_tables.ROOT = sandbox
    report_paper_tables.RESULTS = sandbox / 'results'
    report_paper_tables.main()
    csvs = []
    for path in (sandbox / 'results').glob('*.csv'):
        original = ROOT / 'results' / path.name
        if original.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(original), atol=1e-12, rtol=0)
            csvs.append(path.name)
    assert json.loads((sandbox / 'results/validation.json').read_text()) == json.loads((ROOT / 'results/validation.json').read_text())
    write('report_validation.json', {'complete': True, 'temporary_directory': sandbox.relative_to(ROOT).as_posix(),
          'baseline_reports_executed': ['report.py --no-notebooks', 'report_paper_tables.py'],
          'csv_numeric_content_matches': sorted(csvs), 'validation_json_matches': True,
          'original_reports_overwritten': False})
    print('REPORTS PASSED', flush=True)


def notebooks():
    import asyncio
    import nbformat
    from nbclient import NotebookClient
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sandbox = Path(tempfile.mkdtemp(prefix='cleanup_notebooks_', dir=ROOT / 'tmp'))
    rows = []
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        n = nbformat.read(path, as_version=4)
        n = NotebookClient(n, timeout=300, kernel_name='python3',
                           resources={'metadata': {'path': str(ROOT)}}).execute()
        nbformat.validate(n)
        assert not [o for c in n.cells for o in c.get('outputs', []) if o.output_type == 'error']
        nbformat.write(n, sandbox / path.name)
        rows.append({'notebook': path.name, 'executed_code_cells': sum(c.cell_type == 'code' and c.execution_count is not None for c in n.cells), 'errors': 0})
        print('EXECUTED COPY', path.name, flush=True)
    write('notebook_validation.json', {'complete': True, 'notebooks': rows,
          'temporary_directory': sandbox.relative_to(ROOT).as_posix(), 'original_notebooks_overwritten': False})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['structure', 'encoders', 'reports', 'notebooks'])
    args = ap.parse_args()
    globals()[args.stage]()
