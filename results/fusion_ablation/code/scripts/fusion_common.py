"""Prespecified fusion grid, immutable provenance and aligned cache loading."""
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from embedding_common import ROOT, sha, write_json

sys.path.insert(0, str(ROOT/'src'))
OUT = ROOT/'results/fusion_ablation'
METRICS = ['accuracy','f1_macro','auc','balanced_accuracy']
CLASSIFIERS = ['LogisticRegression','SVM','LightGBM']
VARIANTS = {'KM':['K','M'], 'KG':['K','G'], 'KC':['K','C'], 'KMG':['K','M','G'],
            'KG_adapter':['K','A'], 'KMG_adapter':['K','M','A']}
BRANCH_OPTIONS = {
    'K':[('none','LightGBM')],
    'M':[('anova_64','LogisticRegression'),('pca_var99','SVM')],
    'G':[('anova_256','LightGBM'),('anova_64','SVM'),('pca_var95','LogisticRegression')],
    'A':[('anova_256','LightGBM'),('anova_64','SVM'),('pca_var95','LogisticRegression')],
    'C':[('none','LogisticRegression'),('pca_var99','LogisticRegression')],
}


def option(branch, spec, clf): return '__'.join((branch,spec,clf))


def candidates():
    grid = {'K_only':dict(method='baseline',variant='K',branches=['K'],options=[option('K','none','LightGBM')])}
    for b,opts in BRANCH_OPTIONS.items():
        if b=='K': continue
        for spec,clf in opts:
            key=option(b,spec,clf)
            grid['single__'+key]=dict(method='single',variant=b,branches=[b],options=[key])
    for variant, branches in VARIANTS.items():
        if not variant.endswith('adapter'):
            for clf in CLASSIFIERS:
                name=f'original__{variant}__{clf}'
                grid[name]=dict(method='original',variant=variant,branches=['O' if b=='G' else b for b in branches],
                                classifier=clf)
        reducers=[list(dict.fromkeys(s for s,c in BRANCH_OPTIONS[b])) for b in branches]
        for specs in itertools.product(*reducers):
            for clf in CLASSIFIERS:
                name='__'.join(['compact',variant,*specs,clf])
                grid[name]=dict(method='compact',variant=variant,branches=branches,specs=list(specs),classifier=clf)
        options=[[option(b,s,c) for s,c in BRANCH_OPTIONS[b]] for b in branches]
        for opts in itertools.product(*options):
            for kind,ridge in [('equal',0.),('late',0.),('late',.1)]:
                name='__'.join(['late',variant,*opts,kind,str(ridge)])
                grid[name]=dict(method='late',variant=variant,branches=branches,options=list(opts),
                                fusion=dict(kind=kind,ridge=ridge))
            for gate,budget in itertools.product((.1,.25,.5),(.25,.5)):
                name='__'.join(['adaptive',variant,*opts,str(gate),str(budget)])
                grid[name]=dict(method='adaptive',variant=variant,branches=branches,options=list(opts),
                                fusion=dict(kind='adaptive',ridge=.1,gate=gate,budget=budget))
    return grid


GRID = candidates()


def load_data():
    arrays={}
    keys={'K':'kinematic','M':'moment','O':'gaf_vit_b_16','C':'cwt_efficientnet_b4'}
    ids=y=None
    for b,key in keys.items():
        with np.load(ROOT/'data/derived'/f'{key}.npz') as d:
            if ids is None: ids,y=d['ids'].astype(str),d['y']
            if not np.array_equal(ids,d['ids'].astype(str)) or not np.array_equal(y,d['y']):
                raise ValueError('Branch participant/label mismatch')
            arrays[b]=d['X']
    meta=json.loads((ROOT/'results/embedding_ablation/embeddings/GAF.json').read_text())
    if meta['participants']!=258 or not meta['frozen'] or meta['label_access']:
        raise ValueError('Expected frozen, label-free GAF cache')
    for b,mode in [('G','channel_concat'),('A','group_adapter')]:
        row=meta['outputs'][mode]
        p=ROOT/'results/embedding_ablation/embeddings'/row['file']
        if sha(p)!=row['sha256']: raise ValueError('Embedding cache hash changed')
        arrays[b]=np.load(p,mmap_mode='r')
    for X in arrays.values():
        if len(X)!=258 or not np.isfinite(X).all(): raise ValueError('Invalid feature cache')
    folds=pd.read_csv(ROOT/'results/cv/fold_assignments.csv')
    if len(set(ids))!=258 or not np.array_equal(ids,folds.participant_id) or not np.array_equal(y,folds.label):
        raise ValueError('Original identity changed')
    splits=list(StratifiedKFold(n_splits=6,shuffle=True,random_state=42).split(ids,y))
    for f,(tr,te) in enumerate(splits):
        if not (folds.fold.iloc[te]==f).all(): raise ValueError('Outer folds changed')
    return arrays,ids,y,folds,splits


def signature():
    from evaluate import classifiers
    import sklearn
    inputs=['data/derived/'+p+'.npz' for p in ['kinematic','moment','gaf_vit_b_16','cwt_efficientnet_b4']]
    inputs+=['results/cv/fold_assignments.csv','results/embedding_ablation/embeddings/GAF.json',
             'results/embedding_ablation/embeddings/GAF__channel_concat.npy',
             'results/embedding_ablation/embeddings/GAF__group_adapter.npy']
    code=['scripts/fusion_common.py','scripts/evaluate_fusion_ablation.py','src/diagramo/fusion.py',
          'scripts/evaluate.py','src/diagramo/kernels.py']
    config=dict(seed=42,n_subjects=258,label='group',outer_folds=6,inner_folds=3,meta_crossfit_folds=3,
        primary='mean inner macro-F1; accuracy, AUC, dimensions, candidate ID tie breaks',
        adaptive_selection='strict inner macro-F1 improvement AND no inner Accuracy decrease; otherwise K_only',
        outer_labels_used_for_selection=False,threshold=.5,base_options=BRANCH_OPTIONS,variants=VARIANTS,grid=GRID,
        input_sha256={p:sha(ROOT/p) for p in inputs},code_sha256={p:sha(ROOT/p) for p in code},
        classifier_params={c:classifiers(42)[c].get_params() for c in CLASSIFIERS},
        versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        prior_cohort_reuse='Design informed by previous CV on this same cohort; not external confirmation',
        SVM='predict labels from estimator for early/base; pooled probabilities use threshold .5; no threshold tuning')
    config=json.loads(json.dumps(config,default=str))
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'settings.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=config: raise ValueError('Changed protocol/source/input: use new output directory')
    else:
        originals={str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
                   for folder in ('results','notebooks') for p in (ROOT/folder).rglob('*')
                   if p.is_file() and not p.is_relative_to(OUT) and p.name!='08_fusion_ablation.ipynb'}
        write_json(OUT/'original_artifacts.json',originals)
        write_json(path,config)
        write_json(OUT/'validation.json',dict(complete=False,stage='initialized'))
        for p in code:
            dest=OUT/'code'/p
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((ROOT/p).read_bytes())
    return sha(path)
