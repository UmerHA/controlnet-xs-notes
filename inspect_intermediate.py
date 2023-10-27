from dataclasses import dataclass
import pickle
import torch

@dataclass
class IntermediateOutput:
    i:int
    stage:str
    msg_:str
    t:torch.tensor

    @property
    def msg(self): return 'h_ctrl' if self.msg_=='h_ctr' else 'temb' if self.msg_=='emb' else self.msg_
    
    @property
    def shape(self): return list(self.t.shape) if hasattr(self.t,'shape') else []

    @property
    def head(self,n=5): return self.t.flatten()[:n] if hasattr(self.t,'flatten') else self.t[:n]

def load_intermediate_outputs(filename):
    with open(filename, 'rb') as f:
        log = pickle.load(f)
    return [IntermediateOutput(*l) for l in log]

def print_metadata(outp, n=5):
    if n<0: outp,n = reversed(outp),-n
    for i,o in enumerate(outp):
        print(o.i, o.stage, o.msg, o.shape)
        if i+1>=n: return
