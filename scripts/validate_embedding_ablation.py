"""Mark complete only after numerics, executed notebook, figures, and tests pass."""
import json
import subprocess
import sys
import nbformat
from embedding_common import ROOT,OUT,sha,write_json


def main():
    path=OUT/'validation.json'
    state=json.loads(path.read_text())
    if not state.get('numerical_validation_passed'): raise RuntimeError('Run numerical report validation first')
    for p,h in state['numerical_artifact_sha256'].items():
        if sha(OUT/p)!=h: raise ValueError('Numerical artifact changed after validation: '+p)
    config=json.loads((OUT/'settings.json').read_text())
    evaluation=json.loads((OUT/'evaluation_settings.json').read_text())
    for mapping in (config['input_sha256'],config['code_sha256'],evaluation['code_sha256']):
        for p,h in mapping.items():
            if sha(ROOT/p)!=h: raise ValueError('Source/input changed after numerical validation: '+p)
    book=nbformat.read(ROOT/'notebooks/07_embedding_ablation.ipynb',as_version=4); nbformat.validate(book)
    cells=[c for c in book.cells if c.cell_type=='code']
    if not all(c.execution_count is not None for c in cells): raise ValueError('Notebook not fully executed')
    if any(o.output_type=='error' for c in cells for o in c.outputs): raise ValueError('Notebook errors')
    executed=json.loads((OUT/'notebook_validation.json').read_text())
    if len(executed)!=1 or executed[0]['notebook']!='07_embedding_ablation.ipynb' or executed[0]['errors']:
        raise ValueError('Notebook validation missing')
    manifest=json.loads((OUT/'figures/manifest.json').read_text())
    if len(manifest)!=7: raise ValueError('Seven required figures missing')
    for item in manifest:
        for ext in ('png','svg'):
            p=OUT/'figures'/(item['name']+'.'+ext)
            if not p.exists() or p.stat().st_size<1000: raise ValueError('Figure missing/empty')
        for p,h in item['source_sha256'].items():
            if sha(OUT/p)!=h: raise ValueError('Figure source changed')
    result=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-v'],
                          cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode: raise RuntimeError('Unit tests failed; see tests.log')
    preserved=json.loads((OUT/'original_artifacts.json').read_text())
    for p,h in preserved.items():
        if sha(ROOT/p)!=h: raise ValueError('Original artifact changed')
    write_json(OUT/'test_validation.json',dict(passed=True,returncode=result.returncode,
        test_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tests').glob('test_*.py')}))
    state.update(complete=True,stage='complete',executed_notebook='notebooks/07_embedding_ablation.ipynb',
                 executed_code_cells=len(cells),figures_png=7,figures_svg=7,unit_tests_passed=True,
                 notebook_sha256=sha(ROOT/'notebooks/07_embedding_ablation.ipynb'),
                 report_sha256=sha(OUT/'REPORT_VI.md'))
    state['reporting_code_sha256']={p:sha(ROOT/p) for p in [
        'scripts/report_embedding_ablation.py','scripts/create_embedding_notebook.py',
        'scripts/benchmark_embedding_ablation.py','scripts/audit_embedding_baseline.py',
        'scripts/validate_embedding_ablation.py']}
    write_json(path,state)
    print('EMBEDDING ABLATION COMPLETE: numerical audit, notebook, seven figures, tests, preserved originals')


if __name__=='__main__': main()
