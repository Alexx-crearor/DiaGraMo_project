"""Joint input/time-metadata logit IG; attention is never used as attribution."""
import numpy as np
import torch

def references(batch,kind='zero_joint'):
    keys=('x','patch_meta','segment_meta');base={k:torch.zeros_like(batch[k])for k in keys}
    if kind=='structure_preserving':
        base['x'][:,7]=batch['x'][:,7]
        base['x'][:,13:]=batch['x'][:,13:]
        base['patch_meta'][:,3]=batch['patch_meta'][:,3];base['segment_meta'][:,3]=batch['segment_meta'][:,3]
    elif kind!='zero_joint':raise ValueError(kind)
    return base

def integrated_gradients(model,batch,kind='zero_joint',steps=64):
    """Physical/segmentation inventory held fixed; differentiate x and both time inputs."""
    model.eval();base=references(batch,kind);keys=tuple(base);delta={k:batch[k]-base[k]for k in keys}
    sums={k:torch.zeros_like(batch[k],dtype=torch.float64)for k in keys}
    nodes,weights=np.polynomial.legendre.leggauss(steps)
    for alpha,weight in zip((nodes+1)/2,weights/2):
        point={k:(base[k]+float(alpha)*delta[k]).detach().requires_grad_(True)for k in keys}
        inp=dict(batch,**point);logit=model(inp).sum()
        gradients=torch.autograd.grad(logit,[point[k]for k in keys],allow_unused=True)
        for k,g in zip(keys,gradients):
            if g is not None:sums[k]+=float(weight)*g.double()
    phi={k:(delta[k].double()*sums[k]).detach().cpu().numpy()for k in keys}
    with torch.no_grad():
        output=float(model(batch).reshape(()));reference=float(model(dict(batch,**base)).reshape(()))
    residual=sum(a.sum()for a in phi.values())-(output-reference)
    return dict(**phi,logit=output,reference=reference,residual=residual,steps=steps,baseline_kind=kind)

def converged_ig(model,batch,kind='zero_joint'):
    for steps in (16,32,64,128,256):
        result=integrated_gradients(model,batch,kind,steps)
        tolerance=.003+.003*abs(result['logit']-result['reference'])
        if abs(result['residual'])<=tolerance:return result
    raise ValueError('Temporal IG did not converge; do not mark XAI complete: '+str(result['residual']))

def task_mass(ig,batch):
    patch=np.abs(ig['x']).sum((1,2))+np.abs(ig['patch_meta']).sum(1)
    segment=np.abs(ig['segment_meta']).sum(1);mass=np.zeros(14)
    for ti in range(14):
        mass[ti]=patch[batch['patch_task'].cpu().numpy()==ti].sum()+segment[batch['segment_task'].cpu().numpy()==ti].sum()
    return mass
