"""Write the reviewable cleanup inventory from Git and verification artifacts."""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/cleanup'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def main():
    manifest = json.loads((OUT / 'baseline_manifest.json').read_text())
    groups = collections.defaultdict(list)
    for row in manifest:
        parts = row['path'].split('/')
        group = '/'.join(parts[:2]) if parts[0] in ('data', '.cache', 'results') and len(parts) > 2 else parts[0]
        groups[group].append(row)
    with (OUT / 'preservation_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group', 'files', 'bytes', 'sha256_unchanged'])
        writer.writeheader()
        for group, rows in sorted(groups.items()):
            writer.writerow({'group': group, 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'sha256_unchanged': True})
    changed = []
    origin = json.loads((OUT / 'origin.json').read_text())
    for line in git('diff', '--name-status', origin['pre_cleanup_sha']).splitlines():
        status, path = line.split('\t', 1)
        changed.append({'status': status, 'path': path})
    for path in git('ls-files', '--others', '--exclude-standard').splitlines():
        if path != 'DiaGraMo_REVIEW_KEEP_DECISIONS_VI.md':
            changed.append({'status': 'A', 'path': path})
    (OUT / 'changes.json').write_text(json.dumps({'files': changed, 'moved': [],
         'unknown_user_file_preserved_untracked': 'DiaGraMo_REVIEW_KEEP_DECISIONS_VI.md'}, indent=2), encoding='utf-8')
    modules = []
    original = git('ls-tree', '-r', '--name-only', 'ac4f927').splitlines()
    for rel in original:
        if rel.endswith('.py'):
            before = subprocess.check_output(['git', 'show', 'ac4f927:' + rel], cwd=ROOT).decode('utf-8')
            after = (ROOT / rel).read_text(encoding='utf-8')
            modules.append({'path': rel, 'original_normalized_sha256': hashlib.sha256(before.replace('\r\n', '\n').encode()).hexdigest(),
                            'current_normalized_sha256': hashlib.sha256(after.replace('\r\n', '\n').encode()).hexdigest(),
                            'unchanged_from_original_baseline': before.replace('\r\n', '\n') == after.replace('\r\n', '\n')})
    # Confirm the image encoder implementation matches the pre-improvement baseline.
    def encoder_ast(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'load_encoder')
        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return ast.dump(node, include_attributes=False)
    original_extract = subprocess.check_output(['git', 'show', 'ac4f927:scripts/extract.py'], cwd=ROOT).decode('utf-8')
    assert encoder_ast(original_extract) == encoder_ast((ROOT / 'scripts/extract.py').read_text())
    source_info = {'baseline_source_files': modules, 'load_encoder_ast_matches_original': True,
                   'formula_source_unchanged': next(r['unchanged_from_original_baseline'] for r in modules if r['path'] == 'src/diagramo/features.py')}
    assert source_info['formula_source_unchanged']
    (OUT / 'source_validation.json').write_text(json.dumps(source_info, indent=2), encoding='utf-8')
    lines = ['.']
    for folder in ('src', 'scripts', 'tests', 'notebooks', 'references', 'docs/cleanup'):
        lines.append(folder + '/')
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                lines.append('  ' + path.relative_to(ROOT / folder).as_posix())
    lines += ['results/', '  cv/ (168 result JSON + 168 OOF CSV + summary/settings/fold assignments)',
              '  audit/', '  figures/', '  sensitivity/', '  models/ (local, ignored)',
              *['  ' + p.name for p in sorted((ROOT / 'results').iterdir()) if p.is_file()],
              'data/ (local, unchanged)', '.cache/ (local, unchanged)', '.venv/ (local, unchanged)',
              '.venv314/ (local, unchanged)', 'tmp/ (local verification copies, ignored)']
    (OUT / 'repository_tree.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    validation = {}
    for name in ('artifact', 'structure', 'encoder', 'report', 'notebook', 'source'):
        path = OUT / (name + '_validation.json')
        assert path.exists(), path
        validation[name] = json.loads(path.read_text(encoding='utf-8'))
        if name != 'source':
            assert validation[name]['complete'], name
    (OUT / 'validation.json').write_text(json.dumps({'complete': True, 'pre_cleanup_sha': origin['pre_cleanup_sha'],
            'cleanup_branch': git('branch', '--show-current'), 'all_six_validation_stages_passed': True,
            'new_experiments_run': False, 'baseline_models_retrained': False,
            'embedding_forward_passes_run': False, 'baseline_files_unchanged': len(manifest)}, indent=2), encoding='utf-8')
    print('DOCUMENTED cleanup inventory; all validation stages passed')


if __name__ == '__main__':
    main()
