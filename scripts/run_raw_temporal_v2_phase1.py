"""Recoverable serial GPU queue; CPU physical probes run independently."""
import subprocess
import time
import psutil
from raw_temporal_v2_common import *


def wait_existing_worker():
    while (LOCAL/'gpu_worker.lock').exists():
        previous=json.loads((LOCAL/'gpu_worker.lock').read_text())
        if not psutil.pid_exists(previous['pid']):break
        if abs(psutil.Process(previous['pid']).create_time()-previous['created'])>.01:break
        time.sleep(5)


def run(script):
    print('QUEUE START',script,flush=True)
    subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,check=True)
    print('QUEUE FINISHED',script,flush=True)


if __name__=='__main__':
    wait_existing_worker()
    run('train_raw_temporal_v2_phase1.py')
    run('synthetic_raw_temporal_v2_pooling.py')
    # Do not race with the independently running probe writer.
    while not (OUT/'phase1_bottleneck/probes/validation.json').exists():time.sleep(5)
    run('validate_raw_temporal_v2_phase1.py')
    run('report_raw_temporal_v2_phase1.py')
