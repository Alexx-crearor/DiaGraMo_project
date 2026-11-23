"""Run bounded test suites separately to avoid historical module-name collisions."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/research_cleanup'
LOCAL = ROOT/'data/research_cleanup_local/test_logs'


def run(stage='before', only=None):
    LOCAL.mkdir(parents=True, exist_ok=True)
    suites = [
        ('baseline', ['tests/test_baseline.py']),
        ('kinematic_xai', ['tests/test_kinematic_xai.py']),
        ('temporal', [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_raw_temporal*.py'))]),
        ('locked_reporting', ['tests/test_locked_temporal_reporting.py']),
        ('data_audit', ['tests/data_audit_v3']),
        ('provenance', ['tests/provenance_verification']),
        ('external', ['tests/external_validation']),
        ('native', ['tests/native_channel_research']),
    ]
    if stage=='after':
        suites.append(('cleanup', ['tests/research_cleanup']))
    previous=OUT/f'TEST_RESULTS_{stage.upper()}.json'
    results=json.loads(previous.read_text(encoding='utf8')) if only and previous.exists() else []
    env=dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
             CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for label, paths in suites:
        if only and label!=only:continue
        cmd=[sys.executable,'-B','-m','pytest','-p','scripts.research_cleanup.safe_tests',*paths,'-q']
        note=''
        if stage=='after' and label=='native':
            cmd += ['-k','not test_protected_inputs_unchanged']
            note='One historical full-tree hash assertion passed BEFORE; after authorized README/CLI edits use new preservation manifest, do not rewrite historical manifest.'
        t=time.perf_counter()
        try:
            r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8',timeout=600)
            text=r.stdout+r.stderr; code=r.returncode
        except subprocess.TimeoutExpired as e:
            text=str(e);code=124
        (LOCAL/f'{stage}_{label}.txt').write_text(text,encoding='utf8')
        summaries=re.findall(r'[^\n]*(?:\d+ passed|\d+ failed|\d+ skipped|\d+ error)[^\n]*',text)
        item=dict(suite=label,stage=stage,returncode=code,seconds=time.perf_counter()-t,
                  summary=summaries[-1].strip() if summaries else 'See ignored local log',note=note)
        old=next((r for r in results if r['suite']==label),None)
        if old:item['previous_attempts']=old.get('previous_attempts',[])+[{k:v for k,v in old.items() if k!='previous_attempts'}]
        results=[r for r in results if r['suite']!=label]+[item]
        (OUT/f'TEST_RESULTS_{stage.upper()}.json').write_text(json.dumps(results,indent=2),encoding='utf8')
        print(json.dumps(item),flush=True)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['before','after'],default='before')
    p.add_argument('--suite',choices=['baseline','kinematic_xai','temporal','locked_reporting','data_audit','provenance','external','native','cleanup'])
    args=p.parse_args();results=run(args.stage,args.suite)
    if any(r['returncode'] for r in results):raise SystemExit(1)
