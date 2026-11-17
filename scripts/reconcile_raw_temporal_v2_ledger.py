"""Plan-summary rows are not individual neural fits; reconcile under GPU lock."""
from raw_temporal_v2_common import *


def reconcile():
    phases={phase:json.loads((OUT/f'phase{phase}_validation.json').read_text())
        for phase in range(5) if (OUT/f'phase{phase}_validation.json').exists()}
    groups={1:['p1_probes','p1_pooling','p1_aggregation','p1_task_specific'],2:['p2_models'],
            3:['p3_masked_ssl'],4:['p4_evaluation','p4_fusion']}
    for phase,names in groups.items():
        if phase not in phases or not phases[phase].get('complete'):continue
        for name in names:
            ledger(name,phase=str(phase),training_status='expanded_to_child_runs',completion_status='completed',
                reason='Plan summary, not an extra fit; completed child runs and validation in phase report',
                artifact_paths=f'results/raw_temporal_v2/phase{phase}_validation.json')


if __name__=='__main__':
    with gpu_worker():reconcile()
