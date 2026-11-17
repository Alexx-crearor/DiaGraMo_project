"""Freeze masked SSL and matched downstream controls after Phase2 validation."""
from raw_temporal_v2_common import *


def labeled_subset(train,y,fraction,seed):
    rng=np.random.default_rng(seed);chosen=[]
    for label in np.unique(y[train]):
        indices=rng.permutation(train[y[train]==label]);number=len(indices) if fraction==1 else max(2,int(round(len(indices)*fraction)))
        chosen.extend(indices[:number])
    return np.asarray(sorted(chosen),dtype=int)


def main():
    assert json.loads((OUT/'phase2_validation.json').read_text())['complete']
    decision=json.loads((OUT/'phase2_temporal/development_decision.json').read_text())
    ids,y,schema,train,val,test=development()
    subsets={f'seed{seed}_fraction{fraction}':ids[labeled_subset(train,y,fraction,seed)].tolist()
        for seed in (42,43,44) for fraction in (.25,.5,1.)}
    config=dict(seeds=[42,43],architecture=decision['selected_architecture_for_ssl'],
        supervised_training=decision['selected_training_settings_for_ssl'],
        pretraining=dict(local_encoder=decision['selected_local_encoder_for_ssl'],epochs=8,minimum_epochs=3,patience=3,
            learning_rate=.001,weight_decay=.0001,gradient_clip=1.,batch_windows=128,masked_span=32,halo=2),
        frozen_readout_grid=dict(linear=[.01,.1,1.],nonlinear=[.1,1.]),
        label_sensitivity=dict(fractions=[.25,.5,1.],frozen_seeds=[42,43,44],neural_seeds=[42,43],
            frozen_encoder_seed=42,frozen_kind='linear',frozen_hyper=.1,
            fixed_unlabeled_train_participants=143,fixed_labeled_validation_participants=72,
            interpretation='Sensitivity to supervised Train label count, conditional on previously developed architecture and fixed72 labeled validation participants; not total annotation-budget efficiency'),
        labeled_subsets=subsets,threshold=.5,selection='Development macroF1, AUC, fewer trainable parameters; no outer Test',
        checkpoint_selection='Reconstruction validation MSE for SSL; validation BCE for supervised neural models',
        source_phase2_validation_sha256=sha(OUT/'phase2_validation.json'),
        source_phase2_decision_sha256=sha(OUT/'phase2_temporal/development_decision.json'),
        split=dict(fit_ids=ids[train].tolist(),validation_ids=ids[val].tolist(),outer_test_excluded=ids[test].tolist()),
        loss='Per-window channel-balanced MSE on masked valid pen-down coordinates/pressure/speed, participant-balanced gradient accumulation; pen-up-only windows counted as ineligible objective',
        baselines='Train-only participant-balanced constant and interpolation between visible valid endpoints outside masked halo',
        contrastive_status='not_scheduled: no validated label-preserving augmentation policy, preserve conservative masked objective')
    path=ROOT/'configs/raw_temporal_v2/phase3.json'
    if path.exists():assert json.loads(path.read_text())==config,'Frozen Phase3 config changed'
    else:write(path,config)
    print('PHASE3 FROZEN',config['architecture'],flush=True)


if __name__=='__main__':main()
