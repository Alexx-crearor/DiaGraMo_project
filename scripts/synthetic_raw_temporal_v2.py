"""Known-label, held-out synthetic bags testing unchanged v1 components.

These fixtures enter at the 21-channel representation level. They test
component competence, not calibrated biomechanics or clinical prediction.
"""
import argparse
import copy
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from raw_temporal_v2_common import *
from diagramo.temporal.encoders import LocalMovement, TCN, TemporalBlock, Pool

CONFIG_PATH='configs/raw_temporal_v2/phase0.json'
CONFIG=json.loads((ROOT/CONFIG_PATH).read_text())
TESTS=('rare_event','event_order','physical_time','weak_bag')


def fixture(kind,count,seed):
    assert count%2==0
    rng=np.random.default_rng(seed);labels=np.tile([0,1],count//2)
    patches=64 if kind in ('event_order','weak_bag') else 1
    maximum=512 if kind=='rare_event' else 128
    x=np.zeros((count,patches,21,maximum),dtype='float32')
    lengths=np.full((count,patches),128,dtype='int64');present=np.ones((count,patches),dtype=bool)
    meta=np.zeros((count,patches,4),dtype='float32');locations=[];physical_dt=[]
    for pair in range(count//2):
        base=rng.normal(0,.08,(patches,21,maximum)).astype('float32')
        base[:,13:16]=1.;base[:,16:]=0.;base[:,5:7]=0.
        if kind=='rare_event':
            length=int(rng.choice([128,256,512]));location=int(rng.integers(0,length-4))
        else:length=128;location=int(rng.integers(0,patches))
        for label in (0,1):
            i=pair*2+label;x[i]=base;lengths[i]=length
            if kind=='rare_event':
                if label:x[i,0,2,location:location+4]+=4.
            elif kind=='event_order':
                # The complete patch multiset is identical for a paired A/B bag.
                # Only two event-containing patches exchange positions.
                first,second=patches//4,patches*3//4
                modified=base.copy();modified[first,0,:]=3.;modified[second,1,:]=3.
                if label:modified[[first,second]]=modified[[second,first]]
                x[i]=modified
            elif kind=='physical_time':
                x[i]=base;dt=.001 if label==0 else .02
                x[i,:,6,:]=np.log1p(dt)
            else:
                present[i,int(rng.integers(16,patches)):]=False
                location=min(location,int(present[i].sum())-1)
                x[i,location,2,48:64]+=4. if label else -4.
            dt=(.001 if label==0 else .02) if kind=='physical_time' else .005
            starts=np.arange(patches)*length*dt;duration=(length-1)*dt
            meta[i]=np.log1p(np.column_stack([starts,np.full(patches,duration),np.full(patches,dt),np.full(patches,length)]))/4
            x[i,:,:,length:]=0.
            locations.append(location);physical_dt.append(dt)
    sample_mask=np.arange(maximum)[None,None,:]<lengths[:,:,None]
    sample_mask&=present[:,:,None]
    x=np.where(sample_mask[:,:,None,:],x,0.)
    # Sufficient-statistics baseline: no sequence order or cross-channel timing.
    features=[]
    for i in range(count):
        values=x[i].transpose(0,2,1)[sample_mask[i]].astype('float64')
        features.append(np.r_[values.mean(0),values.std(0),meta[i,present[i]].mean(0)])
    features=np.round(np.asarray(features),6)
    if kind=='event_order':np.testing.assert_array_equal(features[::2],features[1::2])
    return dict(x=x,length=lengths,present=present,meta=meta,y=labels,features=features,
        event_location=np.asarray(locations),physical_dt=np.asarray(physical_dt))


class ComponentClassifier(torch.nn.Module):
    def __init__(self,family,dim=32):
        super().__init__();self.family=family
        self.local=LocalMovement(21,dim)
        self.time=torch.nn.Sequential(torch.nn.Linear(4,dim),torch.nn.Tanh())
        self.context=TCN(dim,0.) if family=='tcn' else TemporalBlock(dim,0.) if family=='hierarchical' else None
        self.pool=Pool(dim);self.head=torch.nn.Sequential(torch.nn.LayerNorm(dim),torch.nn.Linear(dim,1))
    def forward(self,x,length,present,meta):
        b,p,c,l=x.shape;z=self.local(x.reshape(b*p,c,l),length.reshape(-1)).reshape(b,p,-1)
        z=z+self.time(meta);z=z.masked_fill(~present[:,:,None],0.)
        if isinstance(self.context,TemporalBlock):z,_=self.context(z,present)
        elif self.context is not None:z=self.context(z,present)
        pooled,_=self.pool(z,present)
        return self.head(pooled).flatten()


def tensors(data):
    return {k:torch.as_tensor(data[k],device=DEVICE) for k in ('x','length','present','meta','y')}


def predict(model,data):
    model.eval();values=[]
    with torch.no_grad():
        for at in range(0,len(data['y']),16):
            ix=slice(at,at+16);values.extend(model(*(data[k][ix] for k in ('x','length','present','meta'))).cpu().numpy())
    return np.asarray(values,dtype='float64')


def run(kind,family,seed):
    seed_all(seed);eid=f'p0_synthetic_{kind}_{family}_seed{seed}'
    directory=OUT/'phase0_diagnostics'/eid;directory.mkdir(parents=True,exist_ok=True)
    sig=signature(CONFIG_PATH,['scripts/synthetic_raw_temporal_v2.py'])
    if (directory/'result.json').exists():
        prior=json.loads((directory/'result.json').read_text());assert prior['signature']==sig['signature']
        print('REUSE',eid,flush=True);return prior
    begin(eid,'0',family,sig,hypothesis='Known-label component competence: '+kind,
        input_representation='Synthetic21-channel bags; unchanged v1 LocalMovement/TCN/TemporalBlock/Pool',
        ablation=kind,split='independent synthetic Train/validation/Test generators; no real people',seed=seed)
    write(directory/'signature.json',sig)
    raw=[fixture(kind,CONFIG['synthetic_'+part+'_bags'],seed+offset) for part,offset in [('training',1000),('validation',2000),('test',3000)]]
    train,val,test=map(tensors,raw)
    control=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=seed)).fit(raw[0]['features'],raw[0]['y'])
    control_p=control.predict_proba(raw[2]['features'])[:,1]
    if kind=='event_order':assert abs(accuracy_score(raw[2]['y'],control_p>.5)-.5)<1e-12
    model=ComponentClassifier(family).to(DEVICE);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0.)
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=0;best_state=None;bad=0;history=[]
    if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
    began=time.perf_counter();gradient_norm=[]
    for epoch in range(1,CONFIG['synthetic_maximum_epochs']+1):
        model.train();order=rng.permutation(len(train['y']));losses=[]
        for at in range(0,len(order),16):
            ix=torch.as_tensor(order[at:at+16],device=DEVICE);optimizer.zero_grad(set_to_none=True)
            logits=model(*(train[k][ix] for k in ('x','length','present','meta')))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,train['y'][ix].float());assert torch.isfinite(loss)
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(gn)
            gradient_norm.append(float(gn));optimizer.step();losses.append(float(loss.detach()))
        vlogit=predict(model,val);vprob=1/(1+np.exp(-vlogit));bce=float(np.mean(np.logaddexp(0,vlogit)-raw[1]['y']*vlogit))
        history.append(dict(epoch=epoch,train_bce=float(np.mean(losses)),validation_bce=bce,
            validation_f1_macro=metrics(raw[1]['y'],vprob)['f1_macro'],validation_accuracy=float(accuracy_score(raw[1]['y'],vprob>.5))))
        if bce<best-1e-5:
            best=bce;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if epoch%20==0:
            pd.DataFrame(history).to_csv(directory/'learning_curve.csv',index=False)
            resource_check();print('SYNTHETIC',kind,family,seed,epoch,'valBCE',round(bce,4),flush=True)
        if epoch>=30 and (bad>=CONFIG['synthetic_patience'] or best<.01):break
    model.load_state_dict(best_state);test_logit=predict(model,test);probability=1/(1+np.exp(-test_logit))
    train_logit=predict(model,train);sync();duration=time.perf_counter()-began
    checkpoint=LOCAL/'phase0'/f'{eid}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=best_state,family=family,signature=sig['signature'],seed=seed,kind=kind),checkpoint)
    pd.DataFrame(history).to_csv(directory/'learning_curve.csv',index=False)
    pd.DataFrame(dict(synthetic_participant=np.arange(len(probability)),label=raw[2]['y'],probability=probability,
        length=raw[2]['length'][:,0],bag_size=raw[2]['present'].sum(1),event_location=raw[2]['event_location'],
        physical_dt=raw[2]['physical_dt'])).to_csv(directory/'heldout_predictions.csv',index=False)
    result=dict(complete=True,experiment_id=eid,signature=sig['signature'],test=kind,family=family,seed=seed,
        epochs=epoch,best_epoch=best_epoch,validation_bce=best,
        train_metrics=metrics(raw[0]['y'],1/(1+np.exp(-train_logit))),
        heldout_metrics=metrics(raw[2]['y'],probability),statistics_baseline=metrics(raw[2]['y'],control_p),
        parameters=sum(p.numel() for p in model.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated() if DEVICE=='cuda' else 0,
        train_seconds=duration,gradient_norm_mean=float(np.mean(gradient_norm)),gradient_norm_max=float(np.max(gradient_norm)),
        checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint),
        order_fixture_mean_std_identical=kind=='event_order',
        scope='Synthetic representation-level component test only; not end-to-end physical SVC simulation, full14-task architecture, or real-data generalization')
    write(directory/'result.json',result)
    ledger(eid,training_status='completed',completion_status='completed',best_checkpoint=result['checkpoint'],
        validation_metrics=dict(bce=best),held_out_metrics=result['heldout_metrics'],parameter_count=result['parameters'],
        gpu_peak_memory=result['peak_gpu_bytes'],training_cost=duration,artifact_paths=[directory.relative_to(ROOT).as_posix()],
        scientific_interpretation='Known-label synthetic held-out result; does not establish clinical temporal learning')
    print('SYNTHETIC DONE',eid,result['heldout_metrics'],flush=True)
    del model,optimizer,train,val,test,raw
    if DEVICE=='cuda':torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tests',nargs='+',default=TESTS);parser.add_argument('--models',nargs='+',default=CONFIG['synthetic_models']);parser.add_argument('--seeds',nargs='+',type=int,default=CONFIG['synthetic_seeds']);args=parser.parse_args()
    with gpu_worker():
        for kind in args.tests:
            for family in args.models:
                for seed in args.seeds:run(kind,family,seed)
    print('ALL SYNTHETIC COMPONENT TESTS FINISHED',flush=True)


if __name__=='__main__':main()
