"""Windows checkpoint replacement retries without changing the trainer."""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
import pytest
import session_raw_temporal_v2_lite as session


def test_transient_checkpoint_lock_retries_and_preserves_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(session,'LOCAL',tmp_path)
    folder=tmp_path/'phase4_lite';folder.mkdir();source=folder/'fit.resume.pt.tmp';target=folder/'fit.resume.pt'
    source.write_bytes(b'new complete state');target.write_bytes(b'last safe state')
    original=Path.replace;calls=[]
    def flaky(path,destination):
        calls.append(path)
        if len(calls)<3:raise PermissionError('temporary Windows reader lock')
        return original(path,destination)
    monkeypatch.setattr(Path,'replace',flaky);monkeypatch.setattr(session.time,'sleep',lambda seconds:None)
    with session.checkpoint_io_retry():source.replace(target)
    assert len(calls)==3 and target.read_bytes()==b'new complete state'
    assert Path.replace is flaky


def test_unrelated_paths_do_not_get_checkpoint_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(session,'LOCAL',tmp_path)
    source=tmp_path/'unrelated.tmp';source.write_bytes(b'preserve');calls=[]
    def blocked(path,destination):calls.append(path);raise PermissionError('not this experiment')
    monkeypatch.setattr(Path,'replace',blocked)
    with session.checkpoint_io_retry(),pytest.raises(PermissionError):source.replace(tmp_path/'other')
    assert len(calls)==1 and source.read_bytes()==b'preserve'
