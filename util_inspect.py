import pickle
import torch
from dataclasses import dataclass
from functools import partial

@dataclass
class IntermediateOutput:
    i:int
    stage:str
    msg_:str
    t:torch.tensor

    def __init__(self, *args): self.i,self.stage,self.msg_,self.t = args if len(args)==4 else (args[0],'',*args[1:])
    
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

def print_metadata(outp, n=None):
    if n is None: n=len(outp)
    if n<0: outp,n = reversed(outp),-n
    for i,o in enumerate(outp):
        print(o.i, o.stage, o.msg, o.shape)
        if i+1>=n: return

def have_same_names(i,oc,ol,do_print=True):
    eq = oc.stage+oc.msg==ol.stage+ol.msg
    if do_print: print('cloud: ', oc.stage,oc.msg,'\nlocal:', ol.stage,ol.msg)
    return eq

def have_same_shapes(i,oc,ol,do_print=True,assert_names_match=True):
    if assert_names_match: assert oc.stage+oc.msg==ol.stage+ol.msg
    eq = oc.shape==ol.shape
    if do_print:
        eq_str = 'Equal' if eq else 'Different'
        print('>> ',oc.stage, oc.msg, '\t\t',eq_str)
        print('cloud: ', oc.shape,'\nlocal:', ol.shape)
        print()
    return eq

def have_same_values(i,oc,ol,prec=2, do_print=True,assert_names_match=True):
    if assert_names_match: assert oc.stage+oc.msg==ol.stage+ol.msg
    eq = (oc.head.round(decimals=prec)==ol.head.round(decimals=prec)).all().item()
    if do_print:
        eq_str = 'Equal' if eq else 'Different'
        print('>> ',oc.stage, oc.msg, '\t\t',eq_str)
        print('cloud: ', oc.head,'\nlocal:', ol.head)
        print()
    return eq

step_comments = [
    'applied base.conv_in',
    'applied ctrl.conv_in',
    'added hint in ctrl',
    'added ctrl -> base',
    'FULL DIVIDE'
] + [
    'concatted base -> ctrl',
    'applied base subblock',
    'applied ctrl subblock',
    'added ctrl -> base',
    'DIVIDE'
] * 8 + [
    'concatted base -> ctrl',
    'applied base block',
    'applied ctrl block',
    'added ctrl -> base'
] + [
    'added ctrl enc -> base dec',
    'concatted base enc -> base dec',
    'applied base block',
    'DIVIDE'
] * 8


def fmt_bool(b, fmt_str):
    b_str = 'y' if b else 'n'
    b_str = ('{:'+fmt_str+'}').format(b_str)
    GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
    return GREEN+b_str+RESET if b else RED+b_str+RESET

def divider(l, full=True):
    if full: print('-'*l)
    else: print('- '*(l//2))

def compare_intermediate_results(outp_cloud, outp_local, n=None):
    if n is None: n=max(len(outp_cloud),len(outp_local))
    i,c,l,en,es,ev,d = '-','cloud','local','equal name?','equal shape?','equal values?','mean abs Δ'
    print(f'{i:<2} | {c:<19} | {l:<19} | {en:<11} | {es:<12} | {ev:<13} | {d:<10}')
    i,c,l,en,es,ev,d = '','','','','','prec=2',''
    print(f'{i:<2} | {c:<19} | {l:<19} | {en:<11} | {es:<12} | {ev:^13} | {d:<10}')
    total_len = 2+3+19+3+19+3+11+3+12+3+13+3+10
    print_line = partial(divider, l=total_len)
    print_thin_line = partial(divider, l=total_len, full=False)
    last_stage = ''
    step_comments_ = step_comments.copy()
    for i in range(n):
        c,l=outp_cloud[i],outp_local[i]
        eq_name = have_same_names(i,c,l,do_print=False)
        eq_shape = have_same_shapes(i,c,l,do_print=False)
        eq_vals = have_same_values(i,c,l,do_print=False,prec=2)
        mean_difference = (c.t-l.t).abs().mean()
        
        if c.stage!=last_stage:
                print_line()
                last_stage=c.stage
            
        print(f'{i:<2} | {c.stage:<6} {c.msg:<12} | {l.stage:<6} {l.msg:<12} | ', end='')
        print(fmt_bool(eq_name, '^11')+' | '+fmt_bool(eq_shape, '^12')+' | '+fmt_bool(eq_vals, '^13')+' | ', end='')
        print(f'{mean_difference:>10.5f}', end='')

        if i>=5:
            print(f'   {step_comments_.pop(0)}')
            if step_comments_[0]=='DIVIDE':
                print_thin_line()
                step_comments_.pop(0)
            if step_comments_[0]=='FULL DIVIDE':
                print_line()
                step_comments_.pop(0)
        else:
            print()
