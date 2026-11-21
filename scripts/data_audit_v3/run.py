"""CLI orchestrator. All outputs remain under data_audit_v3; no clinical-model training."""
from __future__ import annotations
import argparse
import re
import subprocess
import time
import importlib.util
from types import ModuleType
from common import *


def local_module(name: str) -> ModuleType:
    if name not in {'phase_abc','phase_d','phase_ef','report','validate'}:raise ValueError('Unknown audit module')
    path=Path(__file__).resolve().parent/(name+'.py')
    spec=importlib.util.spec_from_file_location('diagramo_data_audit_v3_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=path:raise AssertionError('Audit module escaped audit directory')
    return module


def main(phase: str='all') -> None:
    if phase in ['init','all'] and not (OUT/'PROTECTED_BEFORE.csv').exists():initialize()
    if not (OUT/'PROTECTED_BEFORE.csv').exists():raise RuntimeError('Run --phase init first')
    if phase in ['abc','all']:
        local_module('phase_abc').main()
    if phase in ['d','all']:
        local_module('phase_d').main()
    if phase in ['ef','all']:
        local_module('phase_ef').main()
    if phase in ['report','all']:
        local_module('report').main()
    if phase in ['tests','all']:
        started=time.perf_counter();command=[sys.executable,'-m','pytest','tests/data_audit_v3','-q']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (OUT/'TEST_STDOUT.txt').write_text(result.stdout+result.stderr,encoding='utf-8');print(result.stdout,flush=True)
        passed=re.search(r'(\d+) passed',result.stdout)
        write_json('TEST_RESULTS.json',dict(complete=result.returncode==0,returncode=result.returncode,passed=int(passed[1]) if passed else 0,
            command='.venv/Scripts/python.exe -m pytest tests/data_audit_v3 -q',elapsed_seconds=time.perf_counter()-started))
        if result.returncode:raise AssertionError('Audit tests failed')
    if phase in ['validate','all']:
        module=local_module('validate');module.manifest();module.main()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['init','abc','d','ef','report','tests','validate','all'],default='all')
    main(parser.parse_args().phase)
