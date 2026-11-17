"""Matched local readout competence on independent synthetic generators."""
import time
from raw_temporal_v2_common import *
from synthetic_raw_temporal_v2 import fixture,tensors,predict
from diagramo.temporal.encoders import LocalMovement,Pool
from diagramo.temporal_v2.models import LocalReadout

CONFIG_PATH='configs/raw_temporal_v2/phase1_synthetic.json'
CONFIG=json.loads((ROOT/CONFIG_PATH).read_text())


class PoolingClassifier(torch.nn.Module):
    def __init__(self,mode):
        super().__init__();self.local=LocalReadout(LocalMovement(21,32),mode)
        self.time=torch.nn.Sequential(torch.nn.Linear(4,32),torch.nn.Tanh())
        self.pool=Pool(32);self.head=torch.nn.Sequential(torch.nn.LayerNorm(32),torch.nn.Linear(32,1))
    def forward(self,x,length,present,meta):
        b,p,c,l=x.shape;z,valid=self.local.encode(x.reshape(b*p,c,l),length.reshape(-1));k=z.shape[1]
        z=z.reshape(b,p,k,32)+self.time(meta)[:,:,None,:]
        mask=(valid.reshape(b,p,k)&present[:,:,None]).reshape(b,p*k)
        value,_=self.pool(z.reshape(b,p*k,32).masked_fill(~mask[:,:,None],0.),mask)
        return self.head(value).flatten()


def run(kind,mode,seed):
    seed_all(seed);eid=f'p1_synthetic_{kind}_{mode}_seed{seed}'
    dest=OUT/'phase1_bottleneck'/eid;dest.mkdir(parents=True,exist_ok=True)
    sig=signature(CONFIG_PATH,['scripts/synthetic_raw_temporal_v2_pooling.py',
        'scripts/synthetic_raw_temporal_v2.py','src/diagramo/temporal_v2/models.py'])
    if (dest/'result.json').exists():
        result=json.loads((dest/'result.json').read_text());assert result['signature']==sig['signature'];return result
    begin(eid,'1','cnn_'+mode,sig,hypothesis='Local pooling synthetic competence, not real-data generalization',
        input_representation='Synthetic21-channel windows',ablation=mode,split='Independent synthetic Train/validation/Test',seed=seed)
    raw=[fixture(kind,CONFIG[part+'_bags'],seed+offset) for part,offset in [('training',1000),('validation',2000),('test',3000)]]
    train,val,test=map(tensors,raw);model=PoolingClassifier(mode).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=0.)
    rng=np.random.default_rng(seed);best=1e9;best_state=None;stale=0;history=[];start=time.perf_counter()
    if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();order=rng.permutation(len(train['y']));losses=[];gradients=[]
        for at in range(0,len(order),16):
            ix=torch.as_tensor(order[at:at+16],device=DEVICE);opt.zero_grad(set_to_none=True)
            out=model(*(train[k][ix] for k in ('x','length','present','meta')))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(out,train['y'][ix].float());assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
            opt.step();losses.append(float(loss.detach()));gradients.append(float(norm))
        logits=predict(model,val);bce=float(np.mean(np.logaddexp(0,logits)-raw[1]['y']*logits))
        history.append(dict(epoch=epoch,train_bce=float(np.mean(losses)),validation_bce=bce,gradient_norm=float(np.mean(gradients))))
        if bce<best-1e-5:
            best=bce;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
        else:stale+=1
        if epoch>=CONFIG['minimum_epochs'] and (stale>=CONFIG['patience'] or best<.01):break
    model.load_state_dict(best_state);logits=predict(model,test);p=1/(1+np.exp(-logits));sync()
    checkpoint=LOCAL/'phase1'/f'{eid}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=best_state,mode=mode,kind=kind,seed=seed,signature=sig['signature']),checkpoint)
    pd.DataFrame(history).to_csv(dest/'learning_curve.csv',index=False)
    pd.DataFrame(dict(label=raw[2]['y'],probability=p)).to_csv(dest/'heldout_predictions.csv',index=False)
    result=dict(complete=True,signature=sig['signature'],experiment_id=eid,kind=kind,pooling=mode,seed=seed,
        heldout_metrics=metrics(raw[2]['y'],p),best_epoch=best_epoch,epochs=epoch,parameters=sum(p.numel() for p in model.parameters()),
        train_seconds=time.perf_counter()-start,checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha(checkpoint),scope=CONFIG['scope'])
    write(dest/'result.json',result)
    ledger(eid,training_status='completed',completion_status='completed',held_out_metrics=result['heldout_metrics'],
        best_checkpoint=result['checkpoint'],parameter_count=result['parameters'],training_cost=result['train_seconds'],
        scientific_interpretation='Easy synthetic capacity control; ceiling performance cannot establish superiority')
    print('POOL SYNTHETIC',eid,result['heldout_metrics']['f1_macro'],flush=True)


if __name__=='__main__':
    with gpu_worker():
        for kind in CONFIG['tests']:
            for mode in CONFIG['pooling']:
                for seed in CONFIG['seeds']:run(kind,mode,seed)
