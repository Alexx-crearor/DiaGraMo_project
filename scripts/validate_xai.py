"""Mark XAI complete only after numerical, provenance, notebook and test checks."""
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import nbformat
from PIL import Image
from xai_common import ROOT,OUT,initialize,sha,write_json


def main():
    initialize()
    p=OUT/'validation.json'; result=json.loads(p.read_text())
    if not result.get('numerical_validation_passed'): raise ValueError('Run report_xai.py first')
    result.update(complete=False,stage='final_validation_running'); write_json(p,result)
    if result['outer_folds']!=6 or result['outer_test_attribution_rows']!=258 or result['fusion_rows']!=516 or result['permutation_evaluations']!=4680:
        raise ValueError('Incomplete XAI experiment coverage')
    for rel,h in result['numeric_artifact_sha256'].items():
        if sha(OUT/rel)!=h: raise ValueError('Numeric XAI artifact changed: '+rel)
    old=json.loads((OUT/'original_artifacts.json').read_text())
    for rel,h in old.items():
        if sha(ROOT/rel)!=h: raise ValueError('Old artifact changed: '+rel)
    errors={}; reconstructed=0
    for f in range(6):
        r=json.loads((OUT/'folds'/f'fold_{f}.json').read_text())
        if set(r['background_participants'])-set(r['train_participants']): raise ValueError('Test background leakage')
        if set(r['train_participants'])&set(r['test_participants']): raise ValueError('Train/Test overlap')
        if any(v>2e-6 for v in r['shap_errors'].values()): raise ValueError('SHAP additivity error')
        if any(v>1e-12 for v in r['partition_prediction_equivalence'].values()): raise ValueError('Explainer/model mismatch')
        if r['partition_encoding']['boundaries_checked']!=3*r['partition_encoding']['total_unique_cuts']:
            raise ValueError('Split-boundary proof coverage incomplete')
        for name,value in r['shap_errors'].items(): errors[name]=max(errors.get(name,0),value)
        reconstructed+=int(r['reconstructed'])
    mapping=json.loads((OUT/'mapping_audit.json').read_text())
    if len(mapping['sentinels'])!=3 or any(r['max_abs_difference']>1e-6 for r in mapping['sentinels']): raise ValueError('Mapping source audit failed')
    for row in mapping['sentinels']:
        if sha(ROOT/row['source'])!=row['source_sha256']: raise ValueError('Mapping source changed')
    notebook_path=ROOT/'notebooks/09_xai.ipynb'
    notebook=nbformat.read(notebook_path,as_version=4); nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']; outputs=[o for c in code for o in c.get('outputs',[])]
    if not code or any(c.execution_count is None for c in code) or any(o.output_type=='error' for o in outputs):
        raise ValueError('Notebook not fully executed')
    tables=sum('text/html' in o.get('data',{}) and '<table' in o['data']['text/html'] for o in outputs)
    if tables<10: raise ValueError('Missing numeric XAI tables')
    nv=json.loads((OUT/'notebook_validation.json').read_text())
    if len(nv)!=1 or nv[0]['notebook']!=notebook_path.name or nv[0]['errors']!=0: raise ValueError('Notebook execution audit missing')
    manifest=json.loads((OUT/'figures/manifest.json').read_text())
    if len(manifest)!=17 or len({x['name'] for x in manifest})!=17: raise ValueError('Expected 17 figure pairs')
    for row in manifest:
        if row['source_cohort_size']!=258 or row['n_participants'] not in (1,43,86,258): raise ValueError('Figure sample size mismatch')
        for rel,h in row['source_sha256'].items():
            if sha(OUT/rel)!=h: raise ValueError('Figure source changed')
        for ext in ('png','svg'):
            q=OUT/'figures'/(row['name']+'.'+ext)
            if sha(q)!=row['output_sha256'][ext]: raise ValueError('Figure hash changed')
            if ext=='svg': ET.parse(q)
            else:
                with Image.open(q) as im:
                    im.verify()
                with Image.open(q) as im:
                    if min(im.size)<1200 or min(im.info.get('dpi',(0,0)))<299: raise ValueError('Figure resolution below declared 300 DPI')
    command=[sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-v']
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=180)
    log=completed.stdout+completed.stderr; (OUT/'tests.log').write_text(log,encoding='utf-8')
    count=re.search(r'Ran (\d+) tests',log)
    if completed.returncode!=0 or not count or int(count[1])<25: raise ValueError('Unit tests failed')
    deps=subprocess.run([sys.executable,'-m','pip','check'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
    if deps.returncode: raise ValueError('Dependency consistency check failed')
    write_json(OUT/'test_validation.json',dict(passed=True,count=int(count[1]),returncode=completed.returncode,
        pip_check=deps.stdout.strip(),source_sha256={str(q.relative_to(ROOT)).replace('\\','/'):sha(q) for q in (ROOT/'tests').glob('test_*.py')}))
    sources=['scripts/report_xai.py','scripts/plot_xai.py','scripts/create_xai_notebook.py','scripts/validate_xai.py',
             'tests/test_explain.py','requirements-xai.txt']
    for rel in sources:
        dest=OUT/'code'/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/rel).read_bytes())
    result.update(complete=True,stage='complete',unit_tests_passed=True,unit_tests_count=int(count[1]),
        notebook_executed=True,executed_code_cells=len(code),numeric_table_outputs=tables,
        png_figures=17,svg_figures=17,figure_dpi=300,max_shap_additivity_errors=errors,
        reconstructed_folds=reconstructed,diagnostic_attempts_preserved=['attempts/precision_01','attempts/precision_02'],
        notebook_sha256=sha(notebook_path),figure_manifest_sha256=sha(OUT/'figures/manifest.json'),
        analysis_code_sha256={rel:sha(ROOT/rel) for rel in sources},original_artifacts_unchanged=len(old))
    write_json(p,result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2),flush=True)


if __name__=='__main__': main()
