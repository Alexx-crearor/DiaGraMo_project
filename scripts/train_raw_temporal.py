"""Nested participant CV of raw CNN/TCN/temporal/hierarchical encoders only."""
import argparse,copy,random,time
import torch
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from raw_temporal_common import *
from diagramo.temporal.dataset import load_person,TrainNormalizer,collate_people
from diagramo.temporal.hierarchical_model import RawTemporalModel

CONFIG=json.loads((ROOT/'configs/raw_temporal.json').read_text())
RUNTIME=json.loads((ROOT/'configs/raw_temporal_runtime.json').read_text())
MICRO=RUNTIME['micro_batch_participants']
DEV='cuda'if torch.cuda.is_available()else'cpu'
torch.set_num_threads(RUNTIME['cpu_threads'])
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def sync():
    if DEV=='cuda':torch.cuda.synchronize()
def statistics():
    with np.load(OUT/'cache/statistics.npz')as a:return a['statistics']
def tensor_person(pid,normalizer,family):return normalizer.transform(load_person(str(OUT/'cache'),pid),DEV,family)
def probability(model,normalizer,ids,family,details=False):
    model.eval();pred=[];sync();began=time.perf_counter()
    with torch.no_grad():
        for start in range(0,len(ids),MICRO):
            batch=collate_people([tensor_person(pid,normalizer,family)for pid in ids[start:start+MICRO]])
            result=model(batch,return_details=details)
            pred.extend(torch.sigmoid(result[0]if details else result).cpu().numpy().tolist())
    sync();return np.asarray(pred),time.perf_counter()-began

def fit(ids,y,fit_idx,family,candidate,seed,val_idx=None,epochs=None):
    seed_all(seed);config=CONFIG['training'];normalizer=TrainNormalizer().fit(statistics(),fit_idx,ids[fit_idx],family=='index_time')
    model=RawTemporalModel(family,**candidate).to(DEV);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss();best=float('inf');best_epoch=0;state=None;bad=0;history=[]
    if DEV=='cuda':torch.cuda.reset_peak_memory_stats()
    sync();began=time.perf_counter();generator=np.random.default_rng(seed)
    for epoch in range(1,(epochs or config['maximum_epochs'])+1):
        model.train();order=generator.permutation(fit_idx);optimizer.zero_grad(set_to_none=True);losses=[];train_pred=[];train_label=[]
        accum=config['gradient_accumulation']
        for k in range(0,len(order),MICRO):
            selected=order[k:k+MICRO];batch=collate_people([tensor_person(ids[index],normalizer,family)for index in selected])
            logit=model(batch);target=torch.tensor(y[selected],device=DEV,dtype=torch.float32)
            loss=criterion(logit,target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite temporal loss')
            # Last partial accumulation uses its actual participant count.
            group_size=min(accum,len(order)-(k//accum)*accum)
            (loss*len(selected)/group_size).backward();losses.extend([float(loss.detach())]*len(selected));train_pred.extend((logit.detach()>0).cpu().numpy());train_label.extend(y[selected])
            if (k+len(selected))%accum==0 or k+len(selected)==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip']);optimizer.step();optimizer.zero_grad(set_to_none=True)
            del batch,logit,loss
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),train_accuracy=float(np.mean(np.asarray(train_pred)==train_label)))
        if val_idx is not None:
            p,_=probability(model,normalizer,ids[val_idx],family);eps=np.finfo(float).eps
            bce=float(-np.mean(y[val_idx]*np.log(np.clip(p,eps,1))+(1-y[val_idx])*np.log(np.clip(1-p,eps,1))))
            row.update(validation_loss=bce,**{'validation_'+key:value for key,value in metrics(y[val_idx],p).items()})
            if bce<best-1e-4:
                best=bce;best_epoch=epoch;bad=0;state={key:v.detach().cpu().clone()for key,v in model.state_dict().items()}
            else:bad+=1
        history.append(row)
        if epoch%5==0:print('EPOCH',family,seed,epoch,'loss',round(row['train_loss'],4),'val',round(row.get('validation_loss',0.),4),flush=True)
        if val_idx is not None and epoch>=config['minimum_epochs']and bad>=config['patience']:break
    if state is not None:model.load_state_dict(state)
    else:best_epoch=int(epochs)
    sync();cost=dict(train_seconds=time.perf_counter()-began,best_epoch=best_epoch,epochs_ran=len(history),
        parameters=sum(p.numel()for p in model.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated()if DEV=='cuda'else 0)
    return model,normalizer,history,cost

def save_fit(destination,ckpt,model,norm,history,metadata,ids,y,evaluation):
    p,timing=probability(model,norm,ids[evaluation],model.family)
    metadata.update(**metrics(y[evaluation],p),inference_seconds=timing,checkpoint=ckpt.relative_to(ROOT).as_posix())
    ckpt.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(state=model.cpu().state_dict(),architecture=model.config,normalizer=norm,metadata=metadata),ckpt)
    metadata['checkpoint_sha256']=sha(ckpt);write(destination.with_suffix('.json'),metadata);write(destination.with_name(destination.name+'_history.json'),history)
    pd.DataFrame(dict(participant_id=ids[evaluation],label=y[evaluation],fold=metadata['fold'],model=metadata['model'],seed=metadata['seed'],probability=p,prediction=(p>.5).astype(int))).to_csv(destination.with_suffix('.oof.csv'),index=False)
    return metadata
def load_checkpoint(path):
    b=torch.load(path,map_location='cpu',weights_only=False);m=RawTemporalModel(**b['architecture']).to(DEV);m.load_state_dict(b['state']);m.eval();return m,b

def signature():
    paths=[ROOT/'configs/raw_temporal.json',ROOT/'configs/raw_temporal_runtime.json',ROOT/'scripts/train_raw_temporal.py',ROOT/'scripts/raw_temporal_common.py']+[ROOT/'src/diagramo/temporal'/name for name in ('__init__.py','preprocessing.py','segmentation.py','dataset.py','encoders.py','hierarchical_model.py')]
    value={p.relative_to(ROOT).as_posix():sha(p)for p in paths};target=OUT/'training_code.json'
    if target.exists():assert json.loads(target.read_text())==value,'Frozen temporal training code changed'
    else:write(target,value)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def outer(f):
    ids,y,schema,folds,splits=cohort();tr,te=splits[f];sig=signature();folder=OUT/'folds'/f'fold_{f}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        assert json.loads((folder/'complete.json').read_text())['signature']==sig
        print('REUSE OUTER',f,flush=True);return
    inner=list(StratifiedKFold(3,shuffle=True,random_state=42).split(tr,y[tr]));rows=[]
    for family in CONFIG['families']:
        for ci,candidate in enumerate(CONFIG['candidates']):
            for j,(a,b)in enumerate(inner):
                path=folder/'inner'/f'{family}_{ci}_{j}';target=path.with_suffix('.json')
                if target.exists():
                    saved=json.loads(target.read_text());assert saved['signature']==sig;rows.append(saved);continue
                fit_idx,val_idx=tr[a],tr[b]
                model,norm,history,cost=fit(ids,y,fit_idx,family,candidate,42,val_idx)
                info=dict(fold=f,model=family,family=family,candidate=ci,inner=j,seed=42,signature=sig,fit_ids=ids[fit_idx].tolist(),evaluation_ids=ids[val_idx].tolist(),**cost)
                rows.append(save_fit(path,OUT/'checkpoints'/f'outer_{f}/inner/{family}_{ci}_{j}.pt',model,norm,history,info,ids,y,val_idx))
            print('INNER',f,family,ci,'DONE',flush=True)
    df=pd.DataFrame(rows);mean=df.groupby(['family','candidate']).agg(f1=('f1_macro','mean'),auc=('auc','mean'),epochs=('best_epoch','median')).reset_index()
    choices={}
    for family in CONFIG['families']:
        row=mean[mean.family==family].sort_values(['f1','auc','candidate'],ascending=[False,False,True]).iloc[0]
        choices[family]=dict(candidate=int(row.candidate),epochs=max(1,int(row.epochs)),inner_f1=float(row.f1))
    winner=mean.sort_values(['f1','auc','candidate','family'],ascending=[False,False,True,True]).iloc[0]
    # Matched temporal/multiscale ablations use the R4 configuration, not outer scores.
    ci=choices['hierarchical']['candidate']
    for family in CONFIG['ablations']:
        ab=[]
        for j,(a,b)in enumerate(inner):
            path=folder/'inner'/f'{family}_{ci}_{j}';target=path.with_suffix('.json')
            if target.exists():
                saved=json.loads(target.read_text());assert saved['signature']==sig;ab.append(saved);continue
            fit_idx,val_idx=tr[a],tr[b];model,norm,history,cost=fit(ids,y,fit_idx,family,CONFIG['candidates'][ci],42,val_idx)
            info=dict(fold=f,model=family,family=family,candidate=ci,inner=j,seed=42,signature=sig,fit_ids=ids[fit_idx].tolist(),evaluation_ids=ids[val_idx].tolist(),**cost)
            ab.append(save_fit(path,OUT/'checkpoints'/f'outer_{f}/inner/{family}_{ci}_{j}.pt',model,norm,history,info,ids,y,val_idx))
        rows.extend(ab);choices[family]=dict(candidate=ci,epochs=max(1,int(np.median([r['best_epoch']for r in ab]))))
        print('ABLATION INNER',f,family,'DONE',flush=True)
    pd.DataFrame(rows).to_csv(folder/'inner_search.csv',index=False)
    write(folder/'selection.json',dict(choices=choices,nested_winner=str(winner.family),candidate=int(winner.candidate),outer_test_labels_used=False))
    for family,choice in choices.items():
        for seed in CONFIG['seeds']if family in CONFIG['families']else[42]:
            path=folder/'outer'/f'{family}_seed_{seed}'
            if path.with_suffix('.json').exists():
                assert json.loads(path.with_suffix('.json').read_text())['signature']==sig
                continue
            model,norm,history,cost=fit(ids,y,tr,family,CONFIG['candidates'][choice['candidate']],seed,epochs=choice['epochs'])
            info=dict(fold=f,model=family,family=family,candidate=choice['candidate'],seed=seed,signature=sig,epochs=choice['epochs'],fit_ids=ids[tr].tolist(),evaluation_ids=ids[te].tolist(),**cost)
            save_fit(path,OUT/'checkpoints'/f'outer_{f}/final/{family}_seed_{seed}.pt',model,norm,history,info,ids,y,te)
            print('OUTER FIT',f,family,seed,'DONE',flush=True)
    write(folder/'complete.json',dict(complete=True,inner_fits=len(rows),outer_fits=16,signature=sig));print('OUTER COMPLETE',f,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--folds',nargs='+',type=int,default=list(range(6)));args=p.parse_args()
    if args.pilot:
        ids,y,_,_,splits=cohort();tr,te=splits[0]
        for family in CONFIG['families']:
            m,n,h,c=fit(ids,y,tr[:8],family,CONFIG['candidates'][0],42,epochs=2);write(OUT/'pilot_batched'/(family+'.json'),dict(**c,fit_ids=ids[tr[:8]].tolist(),outer_test_evaluated=False));print('PILOT',family,c,flush=True)
    else:
        for f in args.folds:outer(f)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
