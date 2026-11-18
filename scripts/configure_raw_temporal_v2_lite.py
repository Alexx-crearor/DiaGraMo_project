"""Freeze a small fixed-recipe comparison using completed development evidence."""
from raw_temporal_v2_lite_common import *


def main():
    LITE.mkdir(parents=True,exist_ok=True);RECOVERY.mkdir(parents=True,exist_ok=True)
    preserved=preserve_snapshot()
    assert preserved['start_commit'].startswith('05b3e28')
    for phase in range(3):assert json.loads((OUT/f'phase{phase}_validation.json').read_text())['complete']
    p1=json.loads((OUT/'phase1_bottleneck/p1_hierarchical_task_to_person_duration/result.json').read_text())
    ref=json.loads((OUT/'phase2_temporal/p2_p1_reference_seed43/result.json').read_text())
    stroke=[json.loads((OUT/f'phase2_temporal/p2_stroke_transformer_seed{s}/result.json').read_text()) for s in (42,43)]
    dependencies=[OUT/f'phase{i}_validation.json' for i in range(3)]+[
        OUT/'phase1_bottleneck/p1_hierarchical_task_to_person_duration/result.json',
        OUT/'phase2_temporal/p2_p1_reference_seed43/result.json',
        *(OUT/f'phase2_temporal/p2_stroke_transformer_seed{s}/result.json' for s in (42,43))]
    cfg=dict(seed=42,participants=258,label='group',outer_folds=6,inner_search=False,
        existing_fold_assignments_sha256=sha(ROOT/'results/cv/fold_assignments.csv'),
        threshold=.5,probability_average_weights=[.5,.5],inference_scope='Exploratory historically inspected cohort',
        primary='duration',execution_order=['duration','stroke_transformer'],
        candidate_selection='Phase1 duration-aware reference exceeds best Phase2 family in two-seed development mean F1; one primary fixed before outer evaluation',
        epoch_rule='Round median of development BCE-best epochs across seeds42,43, without outer-Test selection',
        candidates=dict(duration=dict(architecture=dict(model_kind='recovery',**p1['architecture']),
            fixed_epochs=int(round(np.median([p1['best_epoch'],ref['best_epoch']]))),
            development_seconds_per_epoch=float(np.mean([p1['training_seconds']/p1['epochs_run'],ref['training_seconds']/ref['epochs_run']]))),
            stroke_transformer=dict(architecture=stroke[0]['architecture'],fixed_epochs=int(round(np.median([r['best_epoch'] for r in stroke]))),
            development_seconds_per_epoch=float(np.mean([r['training_seconds']/r['epochs_run'] for r in stroke])))),
        training=dict(learning_rate=.001,weight_decay=.001,gradient_clip=1.,effective_batch=8,micro_batch=2),
        compute=dict(maximum_session_seconds=7200,default_session_minutes=110,report_reserve_seconds=90,
            maximum_gpu_workers=1,gpu_tensor_limit_mib=3500,forecast_safety_factor=1.5,
            full_phase3_budget_seconds=1800,small_ssl_epoch_participants=8,pilot_limit_seconds=900,
            ssl_autolaunch=False,ssl_policy='Recovery smoke/pilot and cost audit only; full suite requires a separate budget-admitted resumable protocol, never launch original unbounded queue'),
        checkpoint='Every optimizer update with optimizer/RNG/order/cursor; evaluation only after fixed final epoch',
        partial_folds='Report Raw, Kinematic and fixed-average on exactly the same held-out participants; no full-cohort claims from partial coverage',
        protocol_dependencies={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies})
    if CONFIG.exists():assert json.loads(CONFIG.read_text())==cfg,'Lite protocol already frozen'
    else:write(CONFIG,cfg)
    write(LITE/'status.json',dict(lite_validation_complete=False,research_complete=False,status='configured',
        full_phase4_status='not_launched_by_request',phase3_status='incomplete_recovery_pending',
        primary_folds_completed=0,secondary_folds_completed=0))
    print('FROZEN LITE: duration',cfg['candidates']['duration']['fixed_epochs'],
          'epochs; Stroke Transformer',cfg['candidates']['stroke_transformer']['fixed_epochs'],'epochs',flush=True)


if __name__=='__main__':main()
