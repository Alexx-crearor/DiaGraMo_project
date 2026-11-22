import numpy as np
import pandas as pd
from scripts.external_validation.external_reader import load_svc_numeric,workbook_participants,xlsx_cells
from .common import SOURCE,sha,read
from .sequence_permutation import permute_rows
def load_data():
 p=pd.read_csv(SOURCE/'external_participant_inventory.csv',dtype={'participant_id':str}).sort_values('participant_id').reset_index(drop=True)
 r=pd.read_csv(SOURCE/'external_recording_inventory.csv',dtype={'participant_id':str})
 assert len(p)==120 and p.participant_id.is_unique and p.label.value_counts().to_dict()=={0:63,1:57}
 source_labels=workbook_participants(xlsx_cells(SOURCE/'data2_SciRep_pub.xlsx')).sort_values('participant_id').reset_index(drop=True)
 assert source_labels.participant_id.tolist()==p.participant_id.tolist() and np.array_equal(source_labels.label,p.label)
 original_oof=pd.read_csv(SOURCE/'prior_oof.csv',dtype={'participant_id':str}).sort_values('participant_id').reset_index(drop=True)
 assert original_oof.participant_id.tolist()==p.participant_id.tolist() and np.array_equal(original_oof.fold,p.candidate_fold) and np.array_equal(original_oof.label,p.label)
 for item in read_source_checksums():assert sha(SOURCE/item['name'])==item['sha256']
 for item in __import__('json').loads((SOURCE/'RAW_RECORDING_CHECKSUMS.json').read_text())['files']:assert sha(SOURCE/item['path'])==item['sha256']
 data=[]
 for row in p.itertuples():
  rows=r[(r.participant_id==row.participant_id)&(r.valid_signal==True)].sort_values('path')
  assert len(rows)>0 and (rows.candidate_fold==row.candidate_fold).all()
  data.append([load_svc_numeric(SOURCE/x) for x in rows.path])
 assert sum(len(x) for rec in data for x in rec)==2176419
 return p,data
def read_source_checksums():
 from .common import ROOT
 import json
 return json.loads((ROOT/'results/external_validation/external_dataset_checksums.json').read_text())['files']
def fit_scaler(data,indices):
 # Equal participants, equal recordings within participants, equal samples within recordings.
 means=[];seconds=[]
 for i in indices:
  means.append(np.mean([x.mean(0) for x in data[i]],0));seconds.append(np.mean([(x*x).mean(0) for x in data[i]],0))
 mean=np.mean(means,0);scale=np.sqrt(np.maximum(np.mean(seconds,0)-mean*mean,1e-12))
 return mean,scale
def windows(recordings,mean,scale,permuted=False,key='',size=256):
 xs=[];ms=[];groups=[];weights=[]
 for r,x in enumerate(recordings):
  x=(x-mean)/scale
  for start in range(0,len(x),size):
   z=x[start:start+size]
   if permuted:z=permute_rows(z,f'{key}:{r}:{start}')
   n=len(z);a=np.zeros((size,7),np.float32);a[:n]=z
   mask=np.zeros(size,np.float32);mask[:n]=1
   xs.append(a.T);ms.append(mask);groups.append(r);weights.append(n)
 return np.stack(xs),np.stack(ms),np.array(groups),np.array(weights)
