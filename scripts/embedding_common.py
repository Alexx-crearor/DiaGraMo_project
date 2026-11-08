"""Frozen, prespecified embedding-ablation protocol and artifact utilities."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/embedding_ablation'
BRANCHES={'GAF':'gaf_vit_b_16','CWT':'cwt_efficientnet_b4'}
ENCODERS={'GAF':'vit_b_16','CWT':'efficientnet_b4'}
MODES=['original','group_adapter','channel_mean','channel_concat']
CLASSIFIERS=['LogisticRegression','SVM','LightGBM']
REDUCERS=[dict(name='none',method='none',value=None),
          *[dict(name=f'pca_{k}',method='pca',value=k) for k in (16,32)],
          *[dict(name=f'pca_var{int(v*100)}',method='pca',value=v) for v in (.95,.99)],
          *[dict(name=f'anova_{k}',method='anova',value=k) for k in (64,256)],
          dict(name='extratrees_256',method='extratrees',value=256)]
PRIMARY=('accuracy','f1_macro','auc')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def initialize():
    """Snapshot artifacts once; never silently mix changed protocol/source/inputs."""
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(seed=42,n_subjects=258,label='group',outer_folds=6,inner_folds=3,
                branches=BRANCHES,encoders=ENCODERS,modes=MODES,classifiers=CLASSIFIERS,
                reducers=REDUCERS,selection='mean inner macro-F1, accuracy, AUC, fewer dimensions, lexical name',
                adapter='fixed convex groups: RGB=(mean(x,y),mean(pressure,pen_status),mean(tilt,azimuth,time))',
                normalization='(image - 0.449) / 0.226; identical to baseline to isolate adaptation',
                encoder_training=False,adapter_training=False,precision='float32',image_size=224,
                input_sha256={p:sha(ROOT/p) for p in ['data/derived/signals512.npy',
                    'data/derived/schema.json','results/cv/fold_assignments.csv',
                    *['data/derived/'+key+'.npz' for key in BRANCHES.values()],
                    'references/weights_manifest.json']},
                checkpoint_sha256={p.name:sha(p) for p in (ROOT/'.cache/torch/hub/checkpoints').glob('*.pth')},
                code_sha256={p:sha(ROOT/p) for p in ['scripts/extract.py','src/diagramo/features.py',
                    'src/diagramo/channel_embedding.py','scripts/embedding_common.py',
                    'scripts/extract_embedding_ablation.py','src/diagramo/reduction.py',
                    'scripts/evaluate.py']},
                versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,
                              sklearn=sklearn.__version__,torch=torch.__version__))
    path=OUT/'settings.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=config:
            raise ValueError('Protocol/source/input changed; do not mix with this extraction')
    else:
        write_json(path,config)
        originals={str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
                   for folder in ('results','notebooks') for p in (ROOT/folder).rglob('*')
                   if p.is_file() and not p.is_relative_to(OUT)}
        write_json(OUT/'original_artifacts.json',originals)
        write_json(OUT/'validation.json',dict(complete=False,stage='initialized',
                   expected_fixed_results=len(BRANCHES)*len(MODES)*len(REDUCERS)*len(CLASSIFIERS)))
    return config
