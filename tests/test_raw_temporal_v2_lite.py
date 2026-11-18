"""Budget enforcement, exact interrupted training, and matched probability rules."""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import pytest
import torch
import raw_temporal_v2_lite_common as common


def test_session_cap_and_reserve(tmp_path,monkeypatch):
    clock=[100.]
    monkeypatch.setattr(common.time,'perf_counter',lambda:clock[0])
    with pytest.raises(AssertionError):common.SessionBudget(121,tmp_path)
    budget=common.SessionBudget(1,tmp_path)
    assert budget.permits(20,reserve=30)
    clock[0]=115.
    assert not budget.permits(20,reserve=30)
    clock[0]=160.
    assert budget.remaining()==0


def test_optimizer_rng_checkpoint_matches_uninterrupted_training(tmp_path):
    # An actual dropout/AdamW training interruption, including all RNG streams.
    common.seed_all(42)
    model=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Dropout(.2),torch.nn.Linear(8,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    generator=np.random.default_rng(42);x=torch.arange(24,dtype=torch.float32).reshape(8,3)/24
    def update(net,opt,rng):
        order=rng.permutation(8);opt.zero_grad(set_to_none=True)
        loss=net(x[order]).square().mean();loss.backward();opt.step()
        return float(loss.detach())
    update(model,optimizer,generator)
    path=tmp_path/'resume.pt'
    common.atomic_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=common.rng_state(generator)),path)
    expected_loss=update(model,optimizer,generator);expected=[p.detach().clone() for p in model.parameters()]
    restored=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Dropout(.2),torch.nn.Linear(8,1))
    other=torch.optim.AdamW(restored.parameters(),lr=.001);saved=torch.load(path,weights_only=False)
    restored.load_state_dict(saved['model']);other.load_state_dict(saved['optimizer']);common.restore_rng(saved['rng'],generator)
    assert update(restored,other,generator)==expected_loss
    for a,b in zip(expected,restored.parameters()):assert torch.equal(a,b)


def test_predefined_average_and_error_categories():
    y=np.array([0,0,1,1]);kin=np.array([.2,.2,.2,.8]);raw=np.array([.2,.8,.8,.2])
    np.testing.assert_allclose(common.fixed_average(kin,raw),[.2,.5,.5,.5])
    assert common.paired_error_categories(y,kin,raw).tolist()==['both_correct','kin_only','raw_only','kin_only']
    with pytest.raises(AssertionError):common.fixed_average(kin,raw[:-1])
    with pytest.raises(AssertionError):common.fixed_average(kin,np.array([.1,np.nan,.3,.4]))
