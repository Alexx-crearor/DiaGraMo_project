"""Held-out inference wrappers, immutable local-token caching and motor IG.

IG explains a conditional motor-input contrast in transformed feature space.
Clock, flags and hierarchy metadata are fixed; its sum is a raw-logit difference,
not a decomposition of the entire probability or a physical counterfactual.
"""
import contextlib
import math
import joblib
from torch.utils.checkpoint import checkpoint
from raw_temporal_v2_common import *
from raw_temporal_v2_training import build_model,tensor
from raw_temporal_v2_evaluation import key
from diagramo.temporal.dataset import collate_people
from diagramo.temporal_v2.multiscale import MultiScaleLocal,OriginalRetainedLocal


class FrozenReadoutTorch(torch.nn.Module):
    """Differentiate the fitted fixed-statistics sklearn readout, without fitting."""
    def __init__(self,encoder_state,pipeline,random_encoder=False,seed=42):
        super().__init__();self.family='cnn'
        if random_encoder:seed_all(seed)
        self.local=MultiScaleLocal() if encoder_state['local_encoder']=='multiscale' else OriginalRetainedLocal()
        if not random_encoder:self.local.load_state_dict(encoder_state['local_state'])
        self.register_buffer('feature_mean',torch.tensor(pipeline[0].mean_,dtype=torch.float64))
        self.register_buffer('feature_scale',torch.tensor(pipeline[0].scale_,dtype=torch.float64))
        estimator=pipeline[-1];self.linear=hasattr(estimator,'coef_')
        weights=[estimator.coef_.T] if self.linear else estimator.coefs_
        biases=[estimator.intercept_] if self.linear else estimator.intercepts_
        self.layer_count=len(weights)
        for i,(weight,bias) in enumerate(zip(weights,biases)):
            self.register_buffer('weight_'+str(i),torch.as_tensor(weight.copy()))
            self.register_buffer('bias_'+str(i),torch.as_tensor(bias.copy()))
    def encode_local(self,x,length):
        tokens=[];valid=[]
        for start in range(0,len(length),128):
            at=slice(start,start+128)
            if torch.is_grad_enabled() and x.requires_grad:
                a,b=checkpoint(self.local,x[at],length[at],use_reentrant=False,preserve_rng_state=False)
            else:a,b=self.local(x[at],length[at])
            tokens.append(a);valid.append(b)
        return torch.cat(tokens),torch.cat(valid)
    def forward(self,batch,patch_present=None,task_present=None,return_details=False):
        tokens,valid=self.encode_local(batch['x'],batch['length'])
        if patch_present is not None:valid=valid&patch_present[:,None]
        if task_present is not None:valid=valid&task_present[batch['patch_task']][:,None]
        features=[];n=batch.get('person_count',1)
        for person in range(n):
            parts=[]
            for task in range(14):
                selected=tokens[valid&(batch['patch_task']==person*14+task)[:,None]]
                if len(selected):parts.extend([selected.mean(0),selected.amax(0),selected.std(0,unbiased=False)])
                else:
                    reference=self.feature_mean[task*96:(task+1)*96].to(tokens.dtype)
                    parts.extend(torch.split(reference,32))
            features.append(torch.cat(parts))
        features=torch.stack(features)
        # sklearn StandardScaler preserves float32 input via two in-place casts.
        z=((features.to(torch.float64)-self.feature_mean).to(torch.float32)/self.feature_scale).to(torch.float32)
        for i in range(self.layer_count):
            weight=getattr(self,'weight_'+str(i));bias=getattr(self,'bias_'+str(i))
            z=z.to(weight.dtype)@weight+bias
            if i+1<self.layer_count:z=torch.tanh(z)
        logits=z.reshape(n)
        return (logits,dict(person=features,window_tokens=tokens,window_token_valid=valid)) if return_details else logits


def load_model(result,train,seed):
    ids,y,schema,folds,splits=cohort();assert set(result['fit_ids'])==set(ids[train])
    path=ROOT/result['checkpoint'];assert sha(path)==result['checkpoint_sha256']
    if path.suffix=='.pt':
        state=torch.load(path,map_location=DEVICE,weights_only=False)
        model=build_model(state['architecture']);model.load_state_dict(state['state']);normalizer=state['normalizer']
    else:
        state=joblib.load(path);assert set(state['fit_ids'])==set(ids[train])
        pre=OUT/'phase4_complementarity'/f'p4_ssl_pre_{key(train)}_seed{seed}/result.json'
        meta=json.loads(pre.read_text());assert meta['checkpoint_sha256']==result['encoder_checkpoint_sha256']
        assert sha(ROOT/meta['checkpoint'])==meta['checkpoint_sha256']
        encoder=torch.load(ROOT/meta['checkpoint'],map_location=DEVICE,weights_only=False)
        assert set(encoder['fit_ids'])==set(ids[train]);normalizer=encoder['normalizer']
        model=FrozenReadoutTorch(encoder,state['pipeline'],result.get('random_encoder',False),seed)
    assert set(normalizer.fit_ids)==set(ids[train])
    model.to(DEVICE).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,normalizer


@contextlib.contextmanager
def cached_local(model,batch):
    """Cache only invariant local encoding; return a fresh validity mask each call."""
    owner=model if hasattr(model,'encode_local') else model.local
    name='encode_local' if owner is model else 'encode';original=getattr(owner,name)
    with torch.no_grad():tokens,valid=original(batch['x'],batch['length'])
    def cached(x,length):
        assert x.data_ptr()==batch['x'].data_ptr() and x.shape==batch['x'].shape and torch.equal(length,batch['length'])
        return tokens,valid.clone()  # MultiScale forward masks validity in-place.
    setattr(owner,name,cached);model._xai_cache_active=True
    try:yield
    finally:setattr(owner,name,original);model._xai_cache_active=False


def motor_reference(batch,kind):
    x=batch['x'];sample=batch['sample_mask'];reference=x.clone();pen=x[:,7,:]>.5
    masks={c:sample for c in (0,1,3,4)};masks[2]=sample&pen
    for c in (8,9,10):masks[c]=sample&(x[:,13,:]>.5)
    for c in (11,12):masks[c]=sample&(x[:,14,:]>.5)
    for channel,mask in masks.items():
        if kind=='train_mean':value=torch.zeros_like(x[:,channel,:])
        else:
            assert kind=='window_mean'
            mean=(x[:,channel,:]*mask).sum(1)/mask.sum(1).clamp_min(1)
            value=mean[:,None].expand_as(x[:,channel,:])
        reference[:,channel,:]=torch.where(mask,value,x[:,channel,:])
    return reference


def integrated_motor_gradients(model,batch,reference,steps=(16,32,64),atol=.01,rtol=.01):
    assert not getattr(model,'_xai_cache_active',False),'Cached local tokens disconnect input gradients'
    x=batch['x'];delta=x-reference
    with torch.no_grad():
        output=float(model(batch).sum());changed=dict(batch,x=reference);baseline=float(model(changed).sum())
    difference=output-baseline;records=[];attribution=None
    for count in steps:
        gradients=torch.zeros_like(x)
        for i in range(count+1):
            value=(reference+delta*(i/count)).detach().requires_grad_(True)
            score=model(dict(batch,x=value)).sum();gradient=torch.autograd.grad(score,value)[0]
            if not torch.isfinite(gradient).all():
                return None,dict(status='failed_nonfinite_gradient',complete=False,raw_logit=output,reference_raw_logit=baseline,steps=count)
            gradients+=gradient.detach()*(.5 if i in (0,count) else 1.)
        attribution=delta*gradients/count;total=float(attribution.double().sum());error=abs(total-difference)
        records.append(dict(steps=count,attribution_sum=total,absolute_completeness_error=error))
        if error<=atol+rtol*abs(difference):break
    passed=records[-1]['absolute_completeness_error']<=atol+rtol*abs(difference)
    return attribution.detach(),dict(complete=passed,status='passed' if passed else 'failed_completeness',
        raw_logit=output,reference_raw_logit=baseline,raw_logit_difference=difference,
        attribution_sum=records[-1]['attribution_sum'],absolute_completeness_error=records[-1]['absolute_completeness_error'],
        steps=records[-1]['steps'],convergence=records,
        precision='Per-value attribution float32; completeness sum float64',
        score_units='Positive-group raw log-odds difference, conditional on unchanged clock/flags/metadata')


@contextlib.contextmanager
def randomized_head(model,seed):
    if isinstance(model,FrozenReadoutTorch):
        weight=getattr(model,'weight_'+str(model.layer_count-1));bias=getattr(model,'bias_'+str(model.layer_count-1));fanin=weight.shape[0]
    else:
        head=model.base.head[-1] if hasattr(model,'base') else model.head[-1]
        weight=head.weight;bias=head.bias;fanin=weight.shape[1]
    prior_weight=weight.detach().clone();prior_bias=bias.detach().clone()
    generator=torch.Generator(device=weight.device).manual_seed(seed)
    with torch.no_grad():
        weight.copy_(torch.randn(weight.shape,device=weight.device,dtype=weight.dtype,generator=generator)/math.sqrt(fanin));bias.zero_()
    try:yield
    finally:
        with torch.no_grad():weight.copy_(prior_weight);bias.copy_(prior_bias)


def task_occlusion(model,batch):
    assert batch.get('person_count',1)==1
    with torch.no_grad():baseline=float(model(batch).sum());values=[]
    with cached_local(model,batch):
        with torch.no_grad():
            assert abs(float(model(batch).sum())-baseline)<1e-5
            for task in range(14):
                present=batch['patch_task']!=task;value=float(model(batch,patch_present=present).sum())
                values.append(baseline-value)
            assert abs(float(model(batch).sum())-baseline)<1e-5,'Occlusion mutated cached validity/state'
    return np.asarray(values),baseline
