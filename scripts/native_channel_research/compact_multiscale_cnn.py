import torch
from torch import nn
class CompactCNN(nn.Module):
 def __init__(self):
  super().__init__();self.branches=nn.ModuleList([nn.Conv1d(7,32,k,padding=k//2) for k in (5,13,29)])
  self.project=nn.Linear(96,32);self.dropout=nn.Dropout(.1);self.head=nn.Linear(32,1)
 def window_encode(self,x,mask):
  # Mask after convolution; no convolution layers follow masked activations.
  h=torch.cat([torch.relu(c(x)) for c in self.branches],1)
  return (h*mask[:,None]).sum(-1)/mask.sum(-1)[:,None]
 def forward(self,x,mask,groups,weights,chunk=64):
  h=torch.cat([self.window_encode(x[i:i+chunk],mask[i:i+chunk]) for i in range(0,len(x),chunk)])
  records=[]
  for g in torch.unique(groups):
   m=groups==g;w=weights[m];records.append((h[m]*w[:,None]).sum(0)/w.sum())
  subject=torch.stack(records).mean(0)
  return self.head(self.dropout(torch.relu(self.project(subject)))).squeeze()
