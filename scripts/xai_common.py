"""Fixed XAI protocol and read-only access to evaluated outer-fold models."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from embedding_common import ROOT,sha,write_json

sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/xai'
FUSION=ROOT/'results/fusion_ablation'
SCENARIOS=['late__KC','adaptive__KC']
N_REPEATS=30
BACKGROUND=100


def load_data():
    arrays={}; ids=y=None
    for b,key in [('K','kinematic'),('C','cwt_efficientnet_b4')]:
        with np.load(ROOT/'data/derived'/f'{key}.npz') as d:
            if ids is None: ids,y=d['ids'].astype(str),d['y']
            if not np.array_equal(ids,d['ids'].astype(str)) or not np.array_equal(y,d['y']): raise ValueError('Cache identity changed')
            arrays[b]=d['X']
    folds=pd.read_csv(ROOT/'results/cv/fold_assignments.csv')
    if len(set(ids))!=258 or not np.array_equal(ids,folds.participant_id) or not np.array_equal(y,folds.label):
        raise ValueError('Participant/label identity differs from evaluated cohort')
    if arrays['K'].shape!=(258,1036) or arrays['C'].shape!=(258,25088): raise ValueError('Cache widths changed')
    splits=list(StratifiedKFold(6,shuffle=True,random_state=42).split(ids,y))
    for f,(tr,te) in enumerate(splits):
        if not (folds.fold.iloc[te]==f).all() or set(ids[tr])&set(ids[te]): raise ValueError('Outer folds changed')
    return arrays,ids,y,folds,splits


def initialize():
    import shap,sklearn,lightgbm,scipy
    old=json.loads((FUSION/'validation.json').read_text())
    if not old['complete']: raise ValueError('Fusion experiment must have completed validation')
    config=dict(seed=42,n_subjects=258,label='group',outer_folds=6,model_selection=False,model_training=False,
        primary_explainer=dict(type='TreeExplainer',feature_perturbation='tree_path_dependent',model_output='raw',approximate=False),
        secondary_explainers=['interventional raw','interventional probability'],background_train_rows=BACKGROUND,
        interventional_backend='lossless frozen-tree partition encoding to avoid float32 split-threshold rounding in SHAP 0.49; no change to evaluated model',
        background_selection='numpy default_rng(42).choice(outer train positions,100,replace=False); sorted',
        permutation_repeats=N_REPEATS,permutation_seed='42 + 1000*outer_fold; task blocks first, feature groups second',
        group_definition='exact prefix in features.py; Statistics and Energy remain separate; fft_* -> FFT',
        primary_group_ranking='sum_j mean_abs(SHAP_j); also report mean_abs(sum_j SHAP_j) and per-feature average',
        grouped_values_are_group_shapley=False,permutation_test_labels_used_only_for_scoring=True,
        fusion_scenarios=SCENARIOS,cwt_attribution='independent-reference exact linear decomposition, CWT model log-odds; no signal back-projection',
        local_case_selection='lexicographically first participant in each adaptive outcome category; descriptive only',
        checkpoint_missing='reconstruct only required fixed models and selected combiners in results/xai/reconstructed; verify OOF before attribution',
        versions=dict(python=sys.version,shap=shap.__version__,numpy=np.__version__,sklearn=sklearn.__version__,
                      lightgbm=lightgbm.__version__,scipy=scipy.__version__),
        code_sha256={p:sha(ROOT/p) for p in ['src/diagramo/explain.py','scripts/analyze_xai.py','scripts/xai_common.py',
            'src/diagramo/features.py','src/diagramo/fusion.py','scripts/prepare.py','scripts/evaluate.py']},
        input_sha256={p:sha(ROOT/p) for p in ['data/derived/kinematic.npz','data/derived/cwt_efficientnet_b4.npz',
            'data/derived/schema.json','results/cv/fold_assignments.csv','results/cv/kinematic__LightGBM.oof.csv',
            'results/fusion_ablation/settings.json','results/fusion_ablation/nested_oof.csv',
            *[f'results/fusion_ablation/folds/fold_{f}.json' for f in range(6)],
            *[f'results/fusion_ablation/stages/outer_{f}.json' for f in range(6)]]})
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'settings.json'
    if p.exists():
        if json.loads(p.read_text(encoding='utf-8'))!=config: raise ValueError('XAI protocol/source/input changed; do not mix results')
    else:
        originals={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for folder in ('results','notebooks')
                   for p in (ROOT/folder).rglob('*') if p.is_file() and not p.is_relative_to(OUT)
                   and p.name!='09_xai.ipynb'}
        write_json(OUT/'original_artifacts.json',originals)
        write_json(OUT/'settings.json',config)
        write_json(OUT/'validation.json',dict(complete=False,stage='initialized'))
        for rel in config['code_sha256']:
            dest=OUT/'code'/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/rel).read_bytes())
    return sha(OUT/'settings.json')


def load_models(f,arrays,ids,y,tr):
    """All existing joblib artifacts are local and hash-checked before loading."""
    import joblib
    record=json.loads((FUSION/'folds'/f'fold_{f}.json').read_text())
    audit=json.loads((FUSION/'stages'/f'outer_{f}.json').read_text())
    if record['train_participants']!=ids[tr].tolist() or audit['train_participants']!=ids[tr].tolist():
        raise ValueError('Checkpoint training identity differs')
    p=FUSION/audit['checkpoint']; q=FUSION/record['combiner_checkpoint']
    reconstructed=False
    if p.exists():
        if sha(p)!=audit['checkpoint_sha256']: raise ValueError('Untrusted/changed model checkpoint')
        bundle=joblib.load(p)
    else:
        from diagramo.fusion import BranchBank
        from evaluate import build_pipeline,classifiers
        specs={'K':[('none','LightGBM')],'C':[('none','LogisticRegression'),('pca_var99','LogisticRegression')]}
        banks={}; models={}; classifiers_=classifiers(42)
        for b,opts in specs.items():
            bank=BranchBank().fit(arrays[b][tr],y[tr],ids[tr],list(dict.fromkeys(s for s,c in opts)))
            A=bank.transform_many(arrays[b][tr]); banks[b]=bank
            for s,c in opts:
                models['__'.join((b,s,c))]=build_pipeline(c,classifiers_[c],preprocess=False).fit(A[s],y[tr])
        bundle=dict(banks=banks,models=models,train_participants=ids[tr].tolist())
        dest=OUT/'reconstructed'/f'outer_{f}.joblib'; dest.parent.mkdir(exist_ok=True)
        joblib.dump(bundle,dest,compress=3); reconstructed=True
    if q.exists():
        if sha(q)!=record['combiner_checkpoint_sha256']: raise ValueError('Changed combiner checkpoint')
        combiners=joblib.load(q)
    else:
        from diagramo.fusion import ProbabilityFusion
        settings=json.loads((FUSION/'settings.json').read_text()); models={}
        with np.load(FUSION/'selections'/f'fold_{f}_base_crossfit.npz') as z:
            for scenario in SCENARIOS:
                c=settings['grid'][record['results'][scenario]['candidate']]
                if c['method']=='baseline': continue
                P=np.column_stack([z[o][:,1] for o in c['options']])
                models[scenario]=ProbabilityFusion(**c['fusion']).fit(P,y[tr])
        combiners=dict(models=models,train_participants=ids[tr].tolist())
        dest=OUT/'reconstructed'/f'outer_{f}_combiners.joblib'; dest.parent.mkdir(exist_ok=True)
        joblib.dump(combiners,dest,compress=3); reconstructed=True
    for item in (bundle,combiners):
        if item['train_participants']!=ids[tr].tolist(): raise ValueError('Model training participant leakage')
    for b in ('K','C'):
        if bundle['banks'][b].fit_ids_!=ids[tr].tolist(): raise ValueError('Preprocessing fit participants differ')
    return bundle,combiners,record,audit,reconstructed
