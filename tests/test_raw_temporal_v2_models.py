"""Controlled readout changes preserve baseline and padding semantics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import pytest
from diagramo.temporal.encoders import LocalMovement
from diagramo.temporal.segmentation import groups
from diagramo.temporal_v2.models import LocalReadout,RecoveryModel


def bag():
    torch.manual_seed(1);p=28;s=28;t=np.repeat(np.arange(14),2)
    return dict(x=torch.randn(p,21,128),length=torch.tensor([128,17]*14),
        patch_task=torch.tensor(t),patch_segment=torch.arange(s),
        patch_meta=torch.randn(p,4),segment_meta=torch.randn(s,4),
        segment_task=torch.tensor(t),segment_weight=torch.ones(s),
        stroke_groups=torch.tensor(groups(np.arange(s),16)),stroke_group_segment=torch.arange(s),
        task_groups=torch.tensor(groups(t,32)),task_group_task=torch.arange(14),
        flat_groups=torch.tensor(groups(t,32)),flat_group_task=torch.arange(14),
        token_meta=torch.zeros(p,4,4),patch_duration=torch.ones(p),
        segment_duration=torch.ones(s),task_duration=torch.ones(14),person_count=1)


@pytest.mark.parametrize('family',['cnn','tcn','hierarchical'])
def test_exact_baseline(family):
    torch.set_num_threads(2);model=RecoveryModel(family).eval();b=bag()
    with torch.no_grad():torch.testing.assert_close(model(b),model.base(b),atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('mode',['mean','mean_max','mean_max_std','attention','multi4'])
def test_local_padding_and_gradient(mode):
    torch.manual_seed(2);model=LocalReadout(LocalMovement(21,32),mode)
    x=torch.randn(4,21,128);length=torch.tensor([1,9,65,128]);changed=x.clone()
    for i,n in enumerate(length):changed[i,:,n:]=999
    z,valid=model.encode(x,length);other,_=model.encode(changed,length)
    torch.testing.assert_close(z,other)
    z.square().sum().backward()
    assert model.backbone.first.weight.grad.abs().sum()>0
    assert torch.isfinite(z).all() and valid.any(1).all()


@pytest.mark.parametrize('stage',['window_to_stroke','stroke_to_task','task_to_person'])
@pytest.mark.parametrize('method',['mean','duration','mean_max'])
def test_stage_and_multitoken_finite(stage,method):
    model=RecoveryModel('hierarchical','multi4',stage,method).eval()
    out,details=model(bag(),return_details=True)
    assert torch.isfinite(out).all() and details['task_valid'].all()
    out.sum().backward()
    assert model.local.backbone.first.weight.grad.abs().sum()>0
