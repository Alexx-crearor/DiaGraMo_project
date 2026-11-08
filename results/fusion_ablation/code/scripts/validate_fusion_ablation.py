"""Completion marker requires numerical audit, executed notebook, plots and tests."""
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import nbformat
from PIL import Image
from fusion_common import ROOT,OUT,GRID,signature,sha,write_json


def main():
    signature()
    path=OUT/'validation.json'; result=json.loads(path.read_text())
    if not result.get('numerical_validation_passed'): raise ValueError('Run report_fusion_ablation.py first')
    result.update(complete=False,stage='final_validation_running')
    write_json(path,result)
    if result['candidates']!=len(GRID) or result['stages']!=78 or result['outer_records']!=6:
        raise ValueError('Incomplete experiment coverage')
    for p,h in result['numerical_artifact_sha256'].items():
        if sha(OUT/p)!=h: raise ValueError('Numerical artifact changed: '+p)
    original=json.loads((OUT/'original_artifacts.json').read_text())
    for p,h in original.items():
        if sha(ROOT/p)!=h: raise ValueError('Old artifact changed: '+p)
    for f in range(6):
        outer=json.loads((OUT/'folds'/f'fold_{f}.json').read_text())
        stage=json.loads((OUT/'stages'/f'outer_{f}.json').read_text())
        for key,hashkey,r in [('checkpoint','checkpoint_sha256',stage),('combiner_checkpoint','combiner_checkpoint_sha256',outer)]:
            if sha(OUT/r[key])!=r[hashkey]: raise ValueError('Checkpoint changed')
    notebook_path=ROOT/'notebooks/08_fusion_ablation.ipynb'
    notebook=nbformat.read(notebook_path,as_version=4); nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    if not code or any(c.execution_count is None for c in code): raise ValueError('Notebook must be fully executed')
    outputs=[o for c in code for o in c.get('outputs',[])]
    if any(o.output_type=='error' for o in outputs): raise ValueError('Notebook has error')
    tables=sum('text/html' in o.get('data',{}) for o in outputs)
    if tables<7: raise ValueError('Missing numeric result tables')
    nv=json.loads((OUT/'notebook_validation.json').read_text())
    if len(nv)!=1 or nv[0]['notebook']!=notebook_path.name or nv[0]['errors']!=0: raise ValueError('Notebook execution validation missing')
    manifest=json.loads((OUT/'figures/manifest.json').read_text())
    if len(manifest)!=7: raise ValueError('Expected seven figures')
    for row in manifest:
        if row['n_participants']!=258: raise ValueError('Wrong figure cohort')
        for p,h in row['source_sha256'].items():
            if sha(OUT/p)!=h: raise ValueError('Plot source changed')
        for ext in ('png','svg'):
            p=OUT/'figures'/(row['name']+'.'+ext)
            if sha(p)!=row['files'][ext]: raise ValueError('Figure changed')
            if ext=='png':
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    if min(im.size)<500: raise ValueError('Undersized plot')
            else: ET.parse(p)
    proc=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-v'],
                        cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=180)
    (OUT/'tests.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    match=re.search(r'Ran (\d+) tests',proc.stdout+proc.stderr)
    if proc.returncode!=0 or not match or int(match[1])<17: raise ValueError('Unit tests failed; see tests.log')
    write_json(OUT/'test_validation.json',dict(passed=True,count=int(match[1]),returncode=proc.returncode,
                                             test_source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/'tests').glob('test_*.py')}))
    code_paths=['scripts/report_fusion_ablation.py','scripts/create_fusion_notebook.py','scripts/validate_fusion_ablation.py',
                'tests/test_fusion.py']
    for rel in code_paths:
        dest=OUT/'code'/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/rel).read_bytes())
    result.update(complete=True,stage='complete',unit_tests_passed=True,unit_tests_count=int(match[1]),
                  executed_code_cells=len(code),numeric_table_outputs=tables,png_figures=7,svg_figures=7,
                  notebook_sha256=sha(notebook_path),figure_manifest_sha256=sha(OUT/'figures/manifest.json'),
                  analysis_code_sha256={p:sha(ROOT/p) for p in code_paths},original_artifacts_unchanged=len(original))
    write_json(path,result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2),flush=True)


if __name__=='__main__': main()
