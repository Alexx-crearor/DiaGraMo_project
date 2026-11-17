"""Dependency-gated SSL queue, one GPU worker, no outer Test development."""
import subprocess
import time
from raw_temporal_v2_common import *


def run(name):
    print('QUEUE START',name,flush=True)
    subprocess.run([sys.executable,'scripts/'+name],cwd=ROOT,check=True)
    print('QUEUE FINISHED',name,flush=True)


if __name__=='__main__':
    while not ((OUT/'phase2_validation.json').exists() and (OUT/'PHASE2_REPORT_VI.md').exists()):time.sleep(5)
    assert json.loads((OUT/'phase2_validation.json').read_text())['complete']
    run('configure_raw_temporal_v2_phase3.py')
    run('train_raw_temporal_v2_phase3.py')
    run('validate_raw_temporal_v2_phase3.py')
    run('report_raw_temporal_v2_phase3.py')
    run('notebook_raw_temporal_v2.py')
