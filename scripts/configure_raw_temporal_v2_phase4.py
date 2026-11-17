"""Freeze nested evaluation, strictly cross-fitted fusion and XAI before Test."""
from raw_temporal_v2_common import *


def main():
    assert json.loads((OUT/'phase3_validation.json').read_text())['complete']
    decision=json.loads((OUT/'phase3_ssl/development_decision.json').read_text())
    p3=json.loads((ROOT/'configs/raw_temporal_v2/phase3.json').read_text())
    table=pd.read_csv(OUT/'tables/phase3_ssl_comparison.csv');means=table.groupby('method')[['f1_macro','auc']].mean()
    ssl_method=decision['selected_ssl_method']
    fusion_source='ssl' if tuple(means.loc[ssl_method])>tuple(means.loc['scratch']) else 'raw'
    source_method=ssl_method if fusion_source=='ssl' else 'scratch'
    source_neural='finetune' if fusion_source=='ssl' else 'scratch'
    def epochs(method):
        return max(1,int(round(np.median([json.loads((OUT/'phase3_ssl'/f'p3_{method}_seed{seed}/result.json').read_text())['best_epoch'] for seed in (42,43)]))))
    pretraining_epochs=max(1,int(round(np.median([json.loads((OUT/'phase3_ssl'/f'p3_masked_ssl_seed{seed}/result.json').read_text())['best_epoch'] for seed in (42,43)]))))
    readout_grid=dict(p3['frozen_readout_grid'])
    if ssl_method=='frozen_ssl_linear':
        best=float(decision['selected_hyper']);other=.1 if best!=.1 else .01
        readout_grid['linear']=sorted([best,other])
    config=dict(seed=42,outer_folds=6,inner_folds=3,participants=258,label='group',
        fold_assignments_sha256=sha(ROOT/'results/cv/fold_assignments.csv'),initialization_seeds=[42,43],
        architecture=decision['architecture'],supervised_training=decision['supervised_training'],
        raw_learning_rates=[.001,.003],preferred_ssl_method=ssl_method,
        frozen_readout_grid=readout_grid,random_frozen_control_when_applicable=True,pretraining=decision['pretraining'],
        pretraining_fixed_epochs=pretraining_epochs,
        pretraining_policy='Fit local encoder only on each supplied Train participant set, fixed development-selected epoch count, no evaluation features or labels used for pretraining selection',
        selection='Mean inner macroF1, then mean AUC, then smaller learning rate/linear C or larger nonlinear alpha; final neural epochs median inner BCE-best epochs',
        primary_seed=42,metrics=['accuracy','f1_macro','auc','pr_auc','balanced_accuracy','sensitivity','specificity','brier'],
        threshold_rule='probability >0.5',bootstrap=dict(replicates=2000,seed=42,unit='participant',
            interpretation='Paired conditional OOF bootstrap; trained models fixed, no CV retraining or adaptive-selection uncertainty'),
        fusion=dict(source=fusion_source,development_method=source_method,
            raw_fixed_epochs=epochs(source_neural),raw_learning_rate=decision['supervised_training']['learning_rate'],
            frozen_hyper=decision['selected_hyper'],
            fixed_average_raw_weight=.5,weight_grid=[i/10 for i in range(11)],
            weight_fit_rule='On Train cross-fitted predictions: maximize macroF1, then AUC, then smaller Raw weight (Kinematic anchor)',
            meta_C=.1,meta_features='Train-scaled clipped logit base probabilities, fixed C0.1, no C search',
            kinematic_calibration_control=True,
            nesting='Outer Train215 -> inner Train143/144, validation72/71; within each inner Train obtain subinner3 OOF base probabilities for combiner fitting. Base models in fusion use fixed epoch/hyperparameters so their OOF labels never choose stopping/parameters. Score combiners on inner validation, then fit final combiners on whole outer-Train OOF. Outer Test only final scoring.',
            optional_seed43_sensitivity='Only if at least six neural runs remain; retain seed42-trained combiner, vary final Raw initialization, do not claim independently selected seed43 fusion pipeline'),
        other_cpu_maximum_fits=126,
        subgroup_analysis=dict(fields=['group','diagnosis_dysgraphia'],minimum_n=20,
            rule='Only actual recorded metadata, descriptive error counts; no invented demographics or causal claims'),
        attribution=dict(target='Task occlusion for every prespecified Raw/SSL procedure, including random frozen control when applicable. Detailed target: highest primary OOF macroF1, then AUC, then raw/SSL/random priority, for descriptive posthoc explanations only. Never refit/select predictive hyperparameters, features or thresholds from this choice; retain every evaluated result.',
            score='Positive-class raw logit; sigmoid probability reported separately',
            references=['Train-normalized motor mean with time/flags fixed','Within-window motor mean with time/flags fixed'],
            valid_derivatives_only=True,ig_steps=[16,32,64],ig_completeness_atol=.01,ig_completeness_rtol=.01,
            task_occlusion_all_participants=True,seeds=[42,43],
            detailed_cases='At most one highest-confidence participant per both_correct/kin_only/raw_only/both_wrong category in each outer fold; posthoc explanation selection only',
            maximum_detailed_cases=24,stroke_occlusion_maximum_segments=64,
            stroke_choice='Top32 by absolute IG plus up to32 uniformly spaced segments within most task-occlusion-sensitive task; deduplicate',
            deletion_fractions=[.1,.2,.5],random_deletion_repeats=10,
            deletion_ranking='Descending signed IG support to original predicted group across all segments of selected task; if IG fails, occlusion-support rank within tested segments only; random control uses same eligible segment set',
            randomization=dict(seed=2026,scope='Gaussian fan-in-scaled final output head only; encoder and hidden layers fixed',
                cases='Detailed cases',methods=['task_occlusion','IG first reference']),
            interpretation_gate=dict(minimum_auc=.6,minimum_balanced_accuracy=.55,
                minimum_seed_task_rank_median_spearman=.3,randomization_similarity_warning=.9,
                scope='Prespecified diagnostic heuristics, not clinical quality thresholds; passing never establishes clinical or causal meaning'),
            time_weight_stress='For duration pooling report uniform-duration counterfactual with fitted parameters fixed; not causal or a retrained ablation',
            failure_rule='Checkpoint replay failure blocks dependent XAI. Failed/unstable IG or randomization does not support substantive behavioral interpretation; retain valid occlusion and report failed checks.'),
        historical_outer_adaptivity=True,inference_scope='Exploratory internal participant CV, not independent confirmation',
        external_required_status_if_unavailable='BLOCKED_DATA_NOT_AVAILABLE',
        dependencies={p:sha(ROOT/p) for p in ['results/raw_temporal_v2/phase3_validation.json',
            'results/raw_temporal_v2/phase3_ssl/development_decision.json','data/derived/kinematic.npz',
            'results/cv/kinematic__LightGBM.oof.csv','results/cv/settings.json']})
    path=ROOT/'configs/raw_temporal_v2/phase4.json'
    if path.exists():assert json.loads(path.read_text())==config,'Frozen Phase4 configuration changed'
    else:write(path,config)
    print('PHASE4 FROZEN',config['preferred_ssl_method'],'fusion source',fusion_source,flush=True)


if __name__=='__main__':main()
