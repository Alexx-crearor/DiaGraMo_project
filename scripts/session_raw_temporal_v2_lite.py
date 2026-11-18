"""Operational Windows I/O retry around the unchanged, signed lite trainer."""
import argparse
from raw_temporal_v2_lite_common import *
from run_raw_temporal_v2_phase4_lite import main as train_session


@contextlib.contextmanager
def checkpoint_io_retry():
    # Windows readers/antivirus can briefly hold a destination open. Retry only
    # atomic .pt.tmp replacements in this experiment's checkpoint directory.
    original=Path.replace;directory=(LOCAL/'phase4_lite').resolve()
    def replace(path,target):
        eligible=path.parent.resolve()==directory and path.name.endswith('.pt.tmp')
        if not eligible:return original(path,target)
        for attempt in range(100):
            try:return original(path,target)
            except PermissionError:
                if attempt==99:raise
                time.sleep(.05)
    Path.replace=replace
    try:yield
    finally:Path.replace=original


def main(minutes=110,deadline=None):
    assert 0<minutes<=120
    if deadline is not None:minutes=min(minutes,(deadline-time.time())/60)
    if minutes<=3:
        print('Not enough time for a budgeted session; saved checkpoints retained',flush=True);return
    write(LITE/'io_retry_provenance.json',dict(operational_change_only=True,
        wrapper_sha256=sha(Path(__file__)),original_trainer_signature=model_signature()['signature'],
        checkpoint_math_and_training_protocol_unchanged=True,retry_seconds_at_most=5,
        invocation_minutes=minutes,explicit_deadline_unix=deadline))
    with gpu_worker(),awake_during_session(),checkpoint_io_retry():train_session(minutes)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=float,default=110)
    parser.add_argument('--deadline-unix',type=float);args=parser.parse_args();main(args.minutes,args.deadline_unix)
