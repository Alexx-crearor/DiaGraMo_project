import hashlib,json,os,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/native_channel_research'
LOCAL=ROOT/'data/native_channel_research_local'
SOURCE=ROOT/'data/external_datasets/drotar_dobes_2020'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,obj,local=False):
 p=(LOCAL if local else OUT)/name;p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');os.replace(t,p)
def read(name,local=False):return json.loads(((LOCAL if local else OUT)/name).read_text(encoding='utf8'))
def md(name,s):
 OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(s,encoding='utf8')
def table(name,rows,local=False):
 import pandas as pd
 p=(LOCAL if local else OUT)/name;p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.tmp');pd.DataFrame(rows).to_csv(t,index=False);os.replace(t,p)
def status(n,s,**kw):write(f'phase{n}_status.json',dict(phase=n,status=s,utc=time.time(),**kw))
class Budget:
 def __init__(self,hours=8):self.start=time.monotonic();self.limit=hours*3600
 def remaining(self):return self.limit-900-(time.monotonic()-self.start)
 def admit(self,seconds):
  if seconds>self.remaining():raise TimeoutError('Session budget: preserve checkpoint and reserve reports')

from contextlib import contextmanager
@contextmanager
def worker_lock(path=None):
 LOCAL.mkdir(parents=True,exist_ok=True)
 with (Path(path) if path is not None else LOCAL/'worker.lock').open('a+b') as handle:
  handle.seek(0)
  if os.name=='nt':
   import msvcrt
   if handle.read(1)==b'':handle.write(b'0');handle.flush()
   handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl
   fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield
  finally:
   handle.seek(0)
   if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
   else:fcntl.flock(handle,fcntl.LOCK_UN)
