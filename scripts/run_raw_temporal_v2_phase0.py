"""Resumeable Phase 0 queue, one actual GPU training process at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/raw_temporal_v2'


def run(script,*args):
    print('RUN',script,*args,flush=True)
    result=subprocess.run([sys.executable,'-X','utf8','-B','scripts/'+script,*args],cwd=ROOT)
    if result.returncode:
        path=OUT/'phase0_diagnostics/queue_failure.json'
        path.write_text(json.dumps(dict(complete=False,failed_script=script,returncode=result.returncode)),encoding='utf-8')
        raise SystemExit(result.returncode)


def main():
    p=argparse.ArgumentParser();p.add_argument('--report-only',action='store_true');args=p.parse_args()
    while not (OUT/'phase0_diagnostics/source_audit/validation.json').exists():
        print('WAIT source fidelity audit; no expensive dependent training',flush=True);time.sleep(30)
    assert json.loads((OUT/'phase0_diagnostics/source_audit/validation.json').read_text())['complete']
    results=list((OUT/'phase0_diagnostics').glob('p0_tiny_*/result.json'));assert len(results)==6
    assert all(json.loads(path.read_text())['memorized'] for path in results),'Memorization gate failed'
    if not args.report_only:run('synthetic_raw_temporal_v2.py')
    run('validate_raw_temporal_v2_phase0.py')
    run('report_raw_temporal_v2_phase0.py')
    print('PHASE0 FINISHED; inspect diagnostic evidence before freezing Phase1 decisions',flush=True)


if __name__=='__main__':main()
