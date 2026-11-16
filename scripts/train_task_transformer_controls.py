"""Matched controls: same small architecture, different processing/task subset."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from task_transformer_common import ROOT,OUT,CONFIG,data,cache,device,features_for,score,sha,write,training_signature
from train_task_transformer import fit,predict,checkpoint,task_mask


def main():
    config=json.loads((ROOT/'configs/task_transformer_controls.json').read_text())
    signature=training_signature()
    X,y,ids,schema,assignment,oof,splits=data();bank=cache();dev=device()
    for f,(tr,te) in enumerate(splits):
        search=pd.read_csv(OUT/'folds'/f'fold_{f}'/'inner_search.csv')
        ranking=pd.read_csv(OUT/'train_only'/f'outer_{f}/final_train/task_importance.csv')
        for control in config['controls']:
            family,ci,label=control['family'],control['candidate'],control['model']
            target=OUT/'controls'/f'fold_{f}'/(label+'_seed_42.json')
            if target.exists():continue
            candidate=CONFIG['candidates'][ci]
            epochs=max(1,int(search[(search.family==family)&(search.candidate==ci)].best_epoch.median()))
            mask=task_mask(family,candidate,schema,ranking)
            feature=features_for(family,bank)
            aug=[bank['train_'+n][tr] for n in CONFIG['nuisances']] if family=='train_augmented' else None
            model,pre,history,cost=fit(feature[tr],y[tr],candidate,family,mask,42,dev,epochs=epochs,augmentations=aug)
            prob,timing=predict(model,pre,feature[te],mask,dev)
            path=OUT/'checkpoints'/f'controls/fold_{f}/{label}.pt'
            row=dict(fold=f,model=label,family=family,seed=42,candidate=ci,epochs=epochs,
                 signature=signature,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),present=mask.tolist(),
                 **cost,inference_seconds=timing['seconds'],**score(y[te],prob),checkpoint=path.relative_to(ROOT).as_posix())
            checkpoint(path,model,pre,mask,ids[tr],ids[te],candidate,family,42,row)
            row['checkpoint_sha256']=sha(path)
            write(target,row);write(target.with_name(target.stem+'_history.json'),history)
            pd.DataFrame(dict(participant_id=ids[te],label=y[te],fold=f,model=label,seed=42,
                             probability=prob,prediction=(prob>.5).astype(int))).to_csv(target.with_suffix('.oof.csv'),index=False)
        print('MATCHED PROCESSING CONTROLS',f,'DONE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
