"""Budgeted research services; original protocols and artifacts stay immutable."""
import random
import uuid
from raw_temporal_v2_common import *
from raw_temporal_v2_training import tensor,build_model,predict,SOURCES as TRAIN_SOURCES
from diagramo.temporal.dataset import TrainNormalizer,load_person,collate_people

LITE=OUT/'phase4_lite'
RECOVERY=OUT/'phase3_recovery'
CONFIG=ROOT/'configs/raw_temporal_v2/phase4_lite.json'


@contextlib.contextmanager
def awake_during_session():
    """Windows sleep prevention exists only for the bounded worker lifetime."""
    if sys.platform!='win32':
        yield;return
    import ctypes
    kernel=ctypes.windll.kernel32
    assert kernel.SetThreadExecutionState(0x80000001),'Cannot hold bounded execution session awake'
    try:yield
    finally:kernel.SetThreadExecutionState(0x80000000)


def atomic_checkpoint(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)


def rng_state(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),generator=generator.bit_generator.state,
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if DEVICE=='cuda' else None)


def restore_rng(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);generator.bit_generator.state=state['generator']
    torch.set_rng_state(state['torch'].cpu())
    if DEVICE=='cuda':torch.cuda.set_rng_state(state['cuda'].cpu())


class SessionBudget:
    def __init__(self,minutes,root=LITE):
        assert 0<minutes<=120
        self.started=time.time();self.deadline=time.perf_counter()+minutes*60
        self.path=root/'sessions'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.json')
        self.record=dict(status='running',started_at_unix=self.started,maximum_seconds=minutes*60,
                         maximum_gpu_workers=1,phases=[],includes_process_setup=False)
        write(self.path,self.record)
    def remaining(self):return max(0.,self.deadline-time.perf_counter())
    def permits(self,seconds,reserve=90):return self.remaining()>=seconds+reserve
    def finish(self,status,**extra):
        self.record.update(status=status,elapsed_seconds=time.time()-self.started,**extra);write(self.path,self.record)


def model_signature():
    return signature(CONFIG.relative_to(ROOT).as_posix(),TRAIN_SOURCES+[
        'scripts/raw_temporal_v2_lite_common.py','scripts/run_raw_temporal_v2_phase4_lite.py'])


def normalizer_for(train):
    ids,_,_,_,_=cohort()
    with np.load(CACHE/'statistics.npz') as data:statistics=data['statistics']
    return TrainNormalizer().fit(statistics,train,ids[train])


def preserve_snapshot():
    path=LITE/'preserved_artifacts.json'
    if path.exists():return json.loads(path.read_text())
    baseline=json.loads((OLD/'protected_manifest.json').read_text())
    v1=json.loads((OLD/'reproducibility_manifest.json').read_text())
    references={**baseline,**v1['source_sha256'],**v1['artifact_sha256']}
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    files=set(tracked)
    files.update(p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file() and
                 not any(part in ('phase3_recovery','phase4_lite') for part in p.relative_to(OUT).parts))
    actual={name:sha(ROOT/name) for name in sorted(files) if (ROOT/name).is_file()}
    for name,digest in references.items():assert sha(ROOT/name)==digest,name
    record=dict(start_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        files=actual,protected_original_references=references,full_phase4_launched=False)
    write(path,record);return record


def check_preservation():
    record=json.loads((LITE/'preserved_artifacts.json').read_text())
    for name,digest in {**record['files'],**record['protected_original_references']}.items():
        assert sha(ROOT/name)==digest,('Existing artifact changed',name)
    return dict(complete=True,existing_files_checked=len(record['files']),
                protected_references_checked=len(record['protected_original_references']))


def paired_error_categories(y,kin,raw):
    k=(np.asarray(kin)>.5)==y;r=(np.asarray(raw)>.5)==y
    return np.where(k&r,'both_correct',np.where(k,'kin_only',np.where(r,'raw_only','both_wrong')))


def fixed_average(kin,raw):
    k=np.asarray(kin);r=np.asarray(raw)
    assert k.shape==r.shape and np.isfinite(k).all() and np.isfinite(r).all()
    assert ((k>=0)&(k<=1)&(r>=0)&(r<=1)).all()
    return .5*k+.5*r
