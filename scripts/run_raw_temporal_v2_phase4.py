"""Dependency-gated final research queue, recoverable without parallel GPU jobs."""
import subprocess
import time
from raw_temporal_v2_common import *


def run(name,*arguments):
    print('QUEUE START',name,*arguments,flush=True)
    subprocess.run([sys.executable,'scripts/'+name,*arguments],cwd=ROOT,check=True)
    print('QUEUE FINISHED',name,flush=True)


if __name__=='__main__':
    while not ((OUT/'phase3_validation.json').exists() and (OUT/'PHASE3_REPORT_VI.md').exists()):time.sleep(5)
    assert json.loads((OUT/'phase3_validation.json').read_text())['complete']
    run('configure_raw_temporal_v2_phase4.py')
    run('train_raw_temporal_v2_phase4.py')
    run('fusion_raw_temporal_v2.py')
    run('validate_raw_temporal_v2_phase4.py','--models-only')
    run('explain_raw_temporal_v2.py')
    run('validate_raw_temporal_v2_xai.py')
    run('plot_raw_temporal_v2_xai.py')
    run('analyze_raw_temporal_v2_phase4.py')
    run('audit_raw_temporal_v2_external.py')
    run('validate_raw_temporal_v2_phase4.py')
    run('report_raw_temporal_v2_phase4.py')
    run('report_raw_temporal_v2.py')
    run('notebook_raw_temporal_v2.py')
    run('validate_raw_temporal_v2_delivery.py')
    # Refresh saved status after successful delivery validation; opening the
    # notebook still only reads artifacts and never retrains models.
    run('notebook_raw_temporal_v2.py')
    run('validate_raw_temporal_v2_delivery.py','--notebook-refresh-only')
