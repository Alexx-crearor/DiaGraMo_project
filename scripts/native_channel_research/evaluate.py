import numpy as np
from .common import *
def complete_oof(frame):
 if len(frame)!=120 or not frame.participant_id.is_unique or set(frame.fold)!=set(range(5)):raise ValueError('Incomplete OOF cannot produce aggregate benchmark')
 return frame.sort_values('participant_id').reset_index(drop=True)
def finalize(p,cfg,budget):
 from .report import generate
 generate(p,cfg,budget)
