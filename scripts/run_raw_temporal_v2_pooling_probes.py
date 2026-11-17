"""Wait for all matched pooling encoders; then use CPU for retention probes."""
import subprocess
import time
from raw_temporal_v2_common import *

if __name__=='__main__':
    for mode in ('mean','mean_max','mean_max_std','attention','multi4'):
        while not (OUT/'phase1_bottleneck'/f'p1_cnn_local_{mode}/result.json').exists():time.sleep(5)
    subprocess.run([sys.executable,'scripts/probe_raw_temporal_v2_pooling.py'],cwd=ROOT,check=True)
