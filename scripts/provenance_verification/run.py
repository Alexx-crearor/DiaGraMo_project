"""Explicit path imports prevent collisions with historical scripts/report.py."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def load(name):
    if name not in {'verifier','reports'}:raise ValueError('Unapproved module')
    path=HERE/(name+'.py')
    spec=importlib.util.spec_from_file_location('diagramo_phase5_'+name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve()==path
    return module


def main():
    v=load('verifier')
    phase=sys.argv[1] if len(sys.argv)>1 else 'all'
    if phase not in {'all','analyze','reports','tests','validate'}:raise ValueError('Unknown phase')
    if phase in {'all','analyze'}:
        v.initialize()
        print('Protected snapshot ready',flush=True)
        v.inspect_local();v.rds_status();v.privacy_inventory()
        print('Local checks complete',flush=True)
    if phase in {'all','reports'}:load('reports').build(v)
    if phase in {'all','tests'}:
        started=time.perf_counter()
        r=subprocess.run([sys.executable,'-m','pytest','tests/provenance_verification','-q'],cwd=v.ROOT,capture_output=True,text=True,encoding='utf-8')
        (v.OUT/'TEST_STDOUT.txt').write_text(r.stdout+r.stderr,encoding='utf-8')
        v.json_out('TEST_RESULTS.json',dict(returncode=r.returncode,elapsed_seconds=time.perf_counter()-started,output=r.stdout.strip()))
        print(r.stdout,flush=True)
        if r.returncode:raise SystemExit(r.returncode)
    if phase in {'all','validate'}:
        c=v.validate();print(json.dumps({k:c[k] for k in ['technical_complete','protected_files_checked','rds_decoded']},ensure_ascii=True))


if __name__=='__main__':main()
