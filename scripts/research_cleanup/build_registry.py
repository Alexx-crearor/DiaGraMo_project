"""Consolidate existing metric artifacts without fitting or changing predictions."""
from pathlib import Path
import json
import math
import subprocess
import sys
import pandas as pd
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[2]
COLUMNS=['research_stage','experiment_id','model','dataset','participant_count',
         'target_definition','input_modality','architecture','split_protocol','seed',
         'evaluation_scope','metrics_source','macro_f1','accuracy','auc','status',
         'limitations','source_commit','report_path']


@lru_cache(maxsize=None)
def source_commit(path):
    return subprocess.check_output(['git','log','-1','--format=%H','--',path],cwd=ROOT,text=True).strip()


def build():
    rows=[]
    def add(stage,identifier,model,source,report,scope,values=None,n=258,target='group: handwriting-disability grouping',architecture='',modality='',status='COMPLETE',split='Participant stratified6 shuffle seed42',seed=42,limitations='Exploratory internal CV; cohort/source and researcher selection limitations'):
        values=values or {}
        row=dict(research_stage=stage,experiment_id=identifier,model=model,dataset='Drotar_Dobes_2020_Slovak' if target.startswith('Workbook') else 'DiaGraMo_258' if n==258 else 'Development/synthetic; see source',participant_count=n,
            target_definition=target,input_modality=modality,architecture=architecture,
            split_protocol=split,seed=seed,evaluation_scope=scope,metrics_source=source,
            macro_f1=values.get('f1_macro',''),accuracy=values.get('accuracy',''),auc=values.get('auc',''),status=status,
            limitations=limitations,source_commit=source_commit(source),report_path=report)
        rows.append(row)
    baseline='results/cv/summary.csv';original=pd.read_csv(ROOT/baseline)
    for r in original.to_dict('records'):
        name='Kinematic_LightGBM' if r['representation']=='kinematic' and r['classifier']=='LightGBM' else r['representation']+'__'+r['classifier']
        add('Original four-branch reproduction',r['representation']+'__'+r['classifier'],name,baseline,'results/REPORT_VI.md','Historical original reproduction; pooled full OOF',r,architecture=r['classifier'],modality=r['representation'])
    reference=original[(original.representation=='kinematic')&(original.classifier=='LightGBM')].iloc[0].to_dict()
    add('Kinematic XAI','kinematic_task_xai','Kinematic_LightGBM',baseline,'results/kinematic_xai/REPORT_VI.md','Previous reporting milestone; held-out model explanation',reference,modality='1036 handcrafted features',architecture='LightGBM + TreeSHAP/grouped permutation/LOTO',limitations='Attribution explains fitted model; not causal; copy importance not minimum-task proof')
    raw='results/raw_temporal/tables/performance_summary.csv'
    for i,r in enumerate(pd.read_csv(ROOT/raw).to_dict('records')):
        add('Initial Raw Temporal',f'raw_temporal_{i}',str(r.get('model',r.get('family','See source'))),raw,'results/raw_temporal/REPORT_VI.md','Historical exploratory evaluation; scope from source',r,n=r.get('participants',r.get('n',258)),modality='Raw chronological handwriting',architecture=str(r.get('model','See source')),seed=r.get('seed',42))
    ledger='results/raw_temporal_v2/EXPERIMENT_LEDGER.csv'
    for r in pd.read_csv(ROOT/ledger).fillna('').to_dict('records'):
        try: split=json.loads(r['split']) if r.get('split') else {}
        except (ValueError,TypeError): split={}
        try: values=json.loads(r.get('held_out_metrics') or r.get('validation_metrics') or '{}')
        except (ValueError,TypeError): values={}
        if not isinstance(values,dict):values={}
        n=len(split.get('evaluation',split.get('validation',[]))) if isinstance(split,dict) else ''
        state=str(r.get('completion_status') or r.get('training_status') or 'UNKNOWN')
        add('Raw Temporal recovery/ablation',r['experiment_id'],r.get('model') or 'Source/probe/control',ledger,'results/raw_temporal_v2/REVIEW_SNAPSHOT_VI.md','Development/synthetic/partial research; NOT full six-fold benchmark',values,n=n or '',target='Synthetic known label or development group; see source',architecture=str(r.get('model','')),modality=str(r.get('input_representation','')),split='Development/synthetic assignment; exact private IDs withheld',seed=r.get('seed',''),status=state,limitations='Historical ledger status may lag later stages; pilot and development scores must not be presented as cohort OOF')
    lite='results/raw_temporal_v2/phase4_lite/paired_comparison.csv'
    for r in pd.read_csv(ROOT/lite).to_dict('records'):
        complete=bool(r['all_six_folds_complete'])
        add('Lightweight Phase4',f"lite_{r['candidate']}_{r['method']}",f"{r['candidate']}__{r['method']}",lite,'results/raw_temporal_v2/PHASE4_LITE_REPORT_VI.md','Exploratory fixed candidate evaluation',r,n=r['participants'],modality='Raw temporal + protected Kinematic',status='COMPLETE' if complete else 'PARTIAL',limitations='Candidate history already inspected; not full original nested Phase4 queue')
    locked='results/raw_temporal_v2/locked_study/matched_oof_metrics.csv'
    names={'Kinematic':'Kinematic_LightGBM','Temporal':'Stroke_Temporal','Fixed50':'Fixed50_Fusion'}
    for r in pd.read_csv(ROOT/locked).to_dict('records'):
        add('Locked six-fold evaluation','locked_'+r['model'],names[r['model']],locked,'results/raw_temporal_v2/locked_study/LOCKED_RESEARCH_REPORT_VI.md','Full258 matched OOF; frozen final configuration; exploratory cohort',r,architecture='LightGBM' if r['model']=='Kinematic' else 'MultiScaleModel stroke relative attention + task pooling' if r['model']=='Temporal' else '0.5 pK + 0.5 pT',modality='Handcrafted' if r['model']=='Kinematic' else '21 temporal channels' if r['model']=='Temporal' else 'Probabilities')
    source='results/data_audit_v3/metadata_shortcut_metrics.csv'
    for r in pd.read_csv(ROOT/source).to_dict('records'):
        add('Data validity/source audit','source_control_'+r['control'],r['control'],source,'results/data_audit_v3/COMPREHENSIVE_DATA_AUDIT_VI.md','Diagnostic Train-only source mapping; not clinical prediction',r,modality='Source metadata/class prior only',architecture='Train category majority/prior',limitations='Strong association; causal source/device contribution not identifiable')
    add('Dataset provenance','phase5_provenance','No new model','results/provenance_verification/LABEL_VERIFICATION_DECISION.json','results/provenance_verification/DATASET_PROVENANCE_VERIFICATION_VI.md','Document/codebook verification, no new model',n=258,status='TECHNICAL_COMPLETE_CLINICAL_PARTIAL',split='Not a performance evaluation',seed='')
    source='results/external_validation/model_metrics.csv';f=pd.read_csv(ROOT/source)
    prior=dict(zip(f.metric,f.value))
    add('External compatibility','phase6_prior','Slovak_B0_Prior',source,'results/external_validation/EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md','External-cohort internal prior control; frozen transfer NOT_RUN',prior,n=120,target='Workbook DYSGR vs0',modality='No signal',architecture='Train prior',split='Reused participant stratified5 seed42',limitations='Physical/task schema incompatible with exact frozen DiaGraMo transfer')
    add('External compatibility','phase6_frozen_transfer','Frozen DiaGraMo transfer','results/external_validation/TRANSFER_PROTOCOL_DECISION.json','results/external_validation/EXTERNAL_VALIDATION_RESEARCH_REPORT_VI.md','Compatibility only; NOT evaluated',n=120,target='Workbook DYSGR vs0',status='BLOCKED_INVALID_CONTRACT',split='NOT_RUN',limitations='No verified14task/channel/unit contract; no fabricated performance')
    for r in pd.read_csv(ROOT/'results/native_channel_research/MODEL_METRICS.csv').to_dict('records'):
        labels={'B0':'Prior','B1':'Length','B2':'Statistical','B3':'OrderedCNN','B4':'PermutedCNN'}
        add('Native-channel benchmark','slovak_'+r['model'],'Slovak_'+r['model']+'_'+labels[r['model']],'results/native_channel_research/MODEL_METRICS.csv','results/native_channel_research/NATIVE_CHANNEL_TEMPORAL_RESEARCH_REPORT_VI.md','Exploratory internal CV on independent cohort; NOT frozen external transfer',r,n=120,target='Workbook DYSGR vs0',modality='Opaque native c0–c6',architecture=labels[r['model']],split='Reuse Phase6 participant stratified5 seed42',limitations='One main seed; unknown physical schema/site; B2 has differences; B4 can retain implicit position')
    add('SSL recovery','phase3_ssl_pilot','Masked SSL pilot','results/raw_temporal_v2/phase3_recovery/status.json','results/raw_temporal_v2/PHASE3_RECOVERY_VI.md','CUDA smoke + one small development pilot only',n='',status='INCOMPLETE_PILOT_ONLY',target='Self-supervised reconstruction',split='Development Train only',modality='Temporal windows',limitations='Full SSL NOT_RUN; cannot claim embedding benefit')
    target=ROOT/'docs/research/EXPERIMENT_REGISTRY.csv';target.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=COLUMNS).to_csv(target,index=False)
    return rows


if __name__=='__main__':
    rows=build();print(f'Indexed {len(rows)} existing experiment/control records; no models fit')
