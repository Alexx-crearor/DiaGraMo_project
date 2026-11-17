"""Dependency-gated Phase2 queue: no fitting before Phase1 validation/report."""
import subprocess
import time
from raw_temporal_v2_common import *


def run(name):
    print('QUEUE START',name,flush=True)
    subprocess.run([sys.executable,'scripts/'+name],cwd=ROOT,check=True)
    print('QUEUE FINISHED',name,flush=True)


if __name__=='__main__':
    while not ((OUT/'phase1_validation.json').exists() and (OUT/'PHASE1_REPORT_VI.md').exists()):time.sleep(5)
    assert json.loads((OUT/'phase1_validation.json').read_text())['complete']
    run('configure_raw_temporal_v2_phase2.py')
    run('notebook_raw_temporal_v2.py')
    run('train_raw_temporal_v2_phase2.py')
    run('validate_raw_temporal_v2_phase2.py')
    run('report_raw_temporal_v2_phase2.py')
    run('notebook_raw_temporal_v2.py')
