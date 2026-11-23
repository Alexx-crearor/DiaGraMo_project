"""Read-only repository inventory, dependency graph and preservation manifests.

Exact private paths stay in an ignored local manifest. No experiments are run.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/research_cleanup'
LOCAL = ROOT / 'data/research_cleanup_local'
NEW = ('docs/research_cleanup/', 'docs/research/', 'docs/weekly_reports/',
       'scripts/research_cleanup/', 'tests/research_cleanup/')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def table(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(rows[0])
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf8').strip()


def private(name):
    return name.startswith(('data/', '.cache/', '.venv/', '.venv314/', 'tmp/',
                            'results/models/', 'results/kinematic_xai/checkpoints/')) or bool(
        re.search(r'/(cache|checkpoints)/|[A-Z]{2}\d{3,}|(?<![A-Za-z0-9])\d{5,}(?![A-Za-z0-9])', name))


def safe_name(name):
    if private(name):
        return 'PRIVATE:' + name.split('/')[0] + ':' + hashlib.sha256(name.encode()).hexdigest()[:16]
    return name


def category(name):
    if private(name): return 'Private/sensitive'
    if 'kinematic_xai' in name: return 'XAI'
    if 'native_channel' in name: return 'Native-channel benchmark'
    if 'external_validation' in name: return 'External dataset research'
    if 'provenance_verification' in name: return 'Dataset provenance'
    if 'data_audit_v3' in name: return 'Data validity'
    if 'temporal' in name: return 'Temporal research'
    if name.startswith(('src/diagramo/branches/', 'results/cv/', 'results/figures/',
                        'results/audit/', 'results/sensitivity/')) or Path(name).name in {
        'extract.py', 'evaluate.py', 'prepare.py', 'features.py', 'run_all.py', 'test_baseline.py'}:
        return 'Original reproduction'
    if name.startswith('notebooks/'):
        number=Path(name).name[:2]
        return 'Original reproduction' if number in {'01','02','03','04'} else 'XAI' if number=='05' else 'Temporal research'
    if name.startswith('results/'): return 'Historical artifact'
    if name.startswith(('docs/', 'references/')): return 'Historical artifact'
    if name.endswith(('.py', '.txt', '.md', '.json', '.yml', '.yaml')): return 'Shared utility'
    return 'Unknown'


def policy(name):
    if '__pycache__' in name or name.startswith(('.pytest_cache/', 'tmp/', '.venv314/')):
        return 'REVIEW_FOR_REMOVAL'
    if name.startswith(('results/', 'docs/cleanup/')): return 'KEEP_HISTORICAL'
    return 'KEEP_ACTIVE'


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (OUT/'PROTECTED_BEFORE.json').exists():
        raise RuntimeError('Before snapshot already exists; refusing to replace it')
    tracked = set(git('ls-files').splitlines())
    start = dict(head=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                 branches=git('branch', '-a'), status=git('status', '--short'), utc_epoch=time.time())
    all_rows, hashes, public, aggregates = [], [], [], {}
    for base, directories, filenames in os.walk(ROOT):
        directories[:] = [d for d in directories if d != '.git' and not (Path(base)/d).is_symlink()]
        for filename in filenames:
            p = Path(base)/filename
            if p.is_symlink(): continue
            name = p.relative_to(ROOT).as_posix()
            if name.startswith(NEW) or name.startswith('data/research_cleanup_local/'):
                continue
            size = p.stat().st_size
            row = dict(path=name, bytes=size, tracked=name in tracked,
                       category=category(name), policy=policy(name))
            all_rows.append(row)
            environment = name.startswith(('.venv/', '.venv314/'))
            generated = '__pycache__' in name or name.startswith('.pytest_cache/')
            if not environment and not generated:
                hashes.append(dict(path=name, bytes=size, sha256=digest(p)))
            if private(name) or environment or generated:
                group = (name.split('/')[0], row['category'], row['policy'])
                a = aggregates.setdefault(group, dict(path=name.split('/')[0]+'/** [aggregate; private/generated paths withheld]',
                    bytes=0, file_count=0, tracked=False, category=row['category'], policy=row['policy']))
                a['bytes'] += size; a['file_count'] += 1
            else:
                public.append(dict(**row, file_count=1))
    write(LOCAL/'inventory_exact.json', all_rows)
    write(LOCAL/'protected_exact.json', hashes)
    table(OUT/'PROJECT_INVENTORY.csv', public+list(aggregates.values()))
    summary = dict(**start, total_files=len(all_rows), total_bytes=sum(r['bytes'] for r in all_rows),
        tracked_files=sum(r['tracked'] for r in all_rows), hashed_files=len(hashes),
        hashed_bytes=sum(r['bytes'] for r in hashes), exact_paths='Ignored data/research_cleanup_local only',
        exact_manifest_sha256=digest(LOCAL/'protected_exact.json'),
        excluded_from_content_hash='Git internals, Python environments, pycache/pytest cache, newly-created cleanup outputs',
        user_untracked_files=[safe_name(x[3:]) for x in start['status'].splitlines() if x.startswith('?? ')],
        no_deletions=True)
    write(OUT/'PROTECTED_BEFORE.json', summary)
    groups = {}
    for row in hashes:
        if row['bytes']:
            groups.setdefault((row['bytes'], row['sha256']), []).append(row['path'])
    dup = []
    for (size, sha), names in groups.items():
        if len(names) < 2: continue
        for name in names:
            dup.append(dict(path=safe_name(name), bytes=size, sha256=sha, identical_count=len(names),
                decision='KEEP_HISTORICAL' if name.startswith(('results/', 'data/', '.cache/')) else 'REVIEW_FOR_REMOVAL',
                reason='Exact byte duplicate; may be intentional snapshot or scientific provenance',
                dependencies='See DEPENDENCY_MAP and local exact inventory; path use must be checked before removal',
                owner_approval_required=True))
    table(OUT/'DUPLICATE_ARTIFACT_REVIEW.csv', dup, ['path','bytes','sha256','identical_count','decision','reason','dependencies','owner_approval_required'])
    dependencies()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def dependencies():
    imports, errors, notebook_rows, absolute = [], [], [], []
    for directory in ('src', 'scripts', 'tests'):
        for p in sorted((ROOT/directory).rglob('*.py')):
            if '__pycache__' in p.parts: continue
            name = p.relative_to(ROOT).as_posix()
            text = p.read_text(encoding='utf-8-sig')
            try: tree = ast.parse(text)
            except SyntaxError as e:
                errors.append(dict(path=name, line=e.lineno, error=e.msg)); continue
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    imports.extend(dict(caller=name, module=x.name, line=n.lineno) for x in n.names)
                elif isinstance(n, ast.ImportFrom):
                    imports.append(dict(caller=name, module='.'*n.level+(n.module or ''), line=n.lineno))
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(r'[A-Z]:[\\/]', line): absolute.append(dict(path=name, line=i))
    for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
        d=json.loads(p.read_text(encoding='utf8'))
        notebook_rows.append(dict(path=p.relative_to(ROOT).as_posix(), cells=len(d['cells']),
            code_cells=sum(x['cell_type']=='code' for x in d['cells']), format=d.get('nbformat'),
            executed_cells=sum(x.get('execution_count') is not None for x in d['cells']),
            executed_during_cleanup=False))
    table(OUT/'IMPORT_EDGES.csv', imports)
    write(OUT/'IMPORT_DEPENDENCY_CHECKS.json', dict(syntax_errors=errors, import_edges=len(imports),
        python_modules=len(set(r['caller'] for r in imports)), hardcoded_windows_locations=absolute,
        report_module_paths=[p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts').rglob('report.py')],
        method='AST imports; does not execute training or resolve dynamic imports; CLI tests separately'))
    write(OUT/'NOTEBOOK_INVENTORY.json', notebook_rows)


def redact_inventory():
    """Rebuild public names from the immutable local inventory, without rereading data."""
    rows=json.loads((LOCAL/'inventory_exact.json').read_text(encoding='utf8'))
    public=[]; groups={}
    for row in rows:
        name=row['path']
        row=dict(row,category=category(name),policy=policy(name))
        if private(name) or '__pycache__' in name or name.startswith('.pytest_cache/'):
            root=name.split('/')[0]
            key=(root,category(name),row['policy'])
            item=groups.setdefault(key,dict(path=root+'/** [aggregate; private/generated paths withheld]',
                bytes=0,tracked=False,category=category(name),policy=row['policy'],file_count=0))
            item['bytes']+=row['bytes'];item['file_count']+=1
        else:public.append(dict(**row,file_count=1))
    table(OUT/'PROJECT_INVENTORY.csv',public+list(groups.values()))
    # Exact duplicate candidates also retain no sensitive filenames.
    with (OUT/'DUPLICATE_ARTIFACT_REVIEW.csv').open(encoding='utf8',newline='') as f:
        reader=csv.DictReader(f); columns=reader.fieldnames; dup=list(reader)
    for row in dup:
        if not row['path'].startswith('PRIVATE:'):row['path']=safe_name(row['path'])
    table(OUT/'DUPLICATE_ARTIFACT_REVIEW.csv',dup,columns)


def verify():
    rows=json.loads((LOCAL/'protected_exact.json').read_text(encoding='utf8'))
    allowed={'README.md', 'scripts/native_channel_research/run.py', 'tests/native_channel_research/test_native.py'}
    changed=[]; missing=[]
    for row in rows:
        p=ROOT/row['path']
        if not p.exists(): missing.append(safe_name(row['path']))
        elif digest(p)!=row['sha256']: changed.append(dict(path=safe_name(row['path']), authorized=row['path'] in allowed))
    result=dict(checked_files=len(rows), missing=missing, changed=changed,
        unexpected_changes=[r for r in changed if not r['authorized']],
        protected_results_unchanged=not any(r['path'].startswith('results/') for r in changed),
        no_deletions=not missing, no_model_retraining=True,
        private_files_staged=[x for x in git('diff','--cached','--name-only').splitlines() if private(x)],
        before_manifest_unchanged=digest(LOCAL/'protected_exact.json')==json.loads((OUT/'PROTECTED_BEFORE.json').read_text(encoding='utf8'))['exact_manifest_sha256'])
    result['valid']=not missing and not result['unexpected_changes'] and not result['private_files_staged'] and result['before_manifest_unchanged']
    write(OUT/'PRESERVATION_VALIDATION.json',result)
    write(OUT/'PROTECTED_AFTER.json',dict(**result, head=git('rev-parse','HEAD'), utc_epoch=time.time()))
    print(json.dumps(result,indent=2))
    if not result['valid']: raise RuntimeError('Preservation gate failed')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory','dependencies','verify','redact'])
    args=parser.parse_args()
    {'inventory':inventory,'dependencies':dependencies,'verify':verify,'redact':redact_inventory}[args.action]()
