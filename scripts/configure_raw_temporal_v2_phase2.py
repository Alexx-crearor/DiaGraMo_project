"""Freeze a small controlled Phase2 set only after completed Phase1 validation."""
from raw_temporal_v2_common import *
from train_raw_temporal_v2_phase1 import experiments as phase1_experiments


def main():
    status=json.loads((OUT/'phase1_validation.json').read_text());assert status['complete']
    decision=json.loads((OUT/'phase1_bottleneck/development_decision.json').read_text())
    candidates=[]
    for design in ('statistical','mil','stroke_transformer'):
        for seed in (42,43):candidates.append(dict(name=design,seed=seed,architecture=dict(design=design)))
    for name,options in [('no_relative_time',dict(relative_time=False)),('no_relative_spatial',dict(relative_spatial=False)),
        ('no_relative_time_and_space',dict(relative_time=False,relative_spatial=False)),('index_only',dict(index_only=True))]:
        candidates.append(dict(name=name,seed=42,architecture=dict(design='stroke_transformer',**options)))
    candidates.append(dict(name='original_encoder_mil',seed=42,architecture=dict(design='mil',local_encoder='original')))
    p1_result_path=OUT/'phase1_bottleneck'/('p1_'+decision['selected_configuration'])/'result.json'
    p1=json.loads(p1_result_path.read_text())
    p1spec=next(r for r in phase1_experiments() if r['name']==decision['selected_configuration'])
    candidates.append(dict(name='p1_reference',seed=43,architecture=dict(model_kind='recovery',**p1['architecture']),
        training_overrides={key:p1spec[key] for key in ('epochs','learning_rate') if key in p1spec}))
    config=dict(development_outer_fold=0,development_inner_fold=0,candidates=candidates,
        training=dict(epochs=80,minimum_epochs=20,patience=12,learning_rate=.001,weight_decay=.001,gradient_clip=1.),
        threshold=.5,checkpoint_selection='Development-validation BCE only',
        family_selection='Compare primary statistical/MIL/stroke_transformer mean macroF1 over seeds42,43, then AUC, then parameter count; controls single seed are diagnostics',
        ssl_local_encoder_selection='Paired MIL seed42: choose original local encoder only if both macroF1 and AUC exceed multiscale MIL; otherwise multiscale. This development choice requires matched scratch controls in Phase3.',
        ssl_architecture_selection='Compare mean macroF1/AUC over two seeds of Phase1 winner versus best primary Phase2 family; Phase1 seed42 reused and its seed43 is a matched control. Prefer higher F1 then AUC then fewer parameters. If Phase1 wins, preserve its original local backbone. If Phase2 wins, use the paired local encoder decision and matched scratch controls.',
        phase1_reference=dict(path=p1_result_path.relative_to(ROOT).as_posix(),sha256=sha(p1_result_path)),
        phase1_decision=decision,phase1_validation_sha256=sha(OUT/'phase1_validation.json'),
        phase1_report_sha256=sha(OUT/'PHASE1_REPORT_VI.md'),
        rationale='Phase1 map-versus-window probes suggest physical access loss, but local readout changes do not automatically improve prediction. Compare multiscale retained-token statistical pooling versus weak-supervision MIL and add only bounded stroke-relative attention. Original-local MIL isolates encoder change; no task Transformer without later evidence.',
        perturbations=dict(transport='Reverse tokens and all relationship metadata together within bounded stroke groups: invariance control',
            content_only='Reverse stroke content, keep relationship metadata in original slots: metadata-inconsistent OOD stress, not realistic writing or causal proof'),
        outer_test_evaluated=False)
    path=ROOT/'configs/raw_temporal_v2/phase2.json'
    if path.exists():assert json.loads(path.read_text())==config,'Frozen Phase2 config changed'
    else:write(path,config)
    print('PHASE2 FROZEN',len(candidates),'development fits',flush=True)


if __name__=='__main__':main()
