from itertools import chain

def public_attrs(o, contains=''):
    contains = contains if isinstance(contains, list) else [contains]
    return [a for a in dir(o) if not a.startswith('_') and any(c in a for c in contains)]

def cls_name(x): return str(type(x)).split('.')[-1].replace("'>",'')
def print_indented_with_type(lv,x,txt): print('\t'*lv, cls_name(x), str(txt))

def simple_describe(x, lv=0,mode='diffusers'):
    if mode=='cnxs':
        if cls_name(x)=='SpatialTransformer': print_indented_with_type(lv, x, (x.proj_in.in_features, x.proj_out.out_features))
        elif cls_name(x)=='ResBlock': print_indented_with_type(lv, x, (x.channels, x.out_channels))
        elif cls_name(x)=='Downsample': print_indented_with_type(lv, x, (x.op.in_channels, x.op.out_channels))
        elif cls_name(x)=='Upsample': print_indented_with_type(lv, x, (x.channels, x.out_channels))
        elif cls_name(x)=='Conv2d': print_indented_with_type(lv, x, (x.in_channels, x.out_channels))
        elif cls_name(x) in ['TimestepEmbedSequential','ModuleList']:
            print_indented_with_type(lv,x,'')
            for m in x: simple_describe(m, lv=lv+1,mode=mode)
        else: print_indented_with_type(lv,x,'')
    
    elif mode=='diffusers':
        # -- basic blocks
        if cls_name(x)=='Transformer2DModel': print_indented_with_type(lv, x, (x.in_channels, x.out_channels))
        elif cls_name(x)=='ResnetBlock2D': print_indented_with_type(lv, x, (x.in_channels, x.out_channels))
        elif cls_name(x)=='Conv2d': print_indented_with_type(lv, x, (x.in_channels, x.out_channels))
        # -- combined blocks - down
        elif cls_name(x)=='CrossAttnDownBlock2D':
            print_indented_with_type(lv,x,'')
            for m in chain.from_iterable(zip(x.resnets, x.attentions)): simple_describe(m, lv=lv+1,mode=mode)
            if hasattr(x,'downsamplers') and x.downsamplers is not None:
                for m in x.downsamplers: simple_describe(m, lv=lv+1,mode=mode)
        elif cls_name(x)=='DownBlock2D':
            print_indented_with_type(lv,x,'')
            for m in x.resnets: simple_describe(m, lv=lv+1,mode=mode)
            if hasattr(x,'downsamplers') and x.downsamplers is not None:
                for m in x.downsamplers: simple_describe(m, lv=lv+1,mode=mode)
        elif cls_name(x)=='Downsample2D': print_indented_with_type(lv, x, (x.conv.in_channels, x.conv.out_channels))
        # -- combined blocks - mid
        elif cls_name(x)=='UNetMidBlock2DCrossAttn':
            print_indented_with_type(lv,x,'')
            for m in chain.from_iterable(zip(x.resnets, x.attentions)): simple_describe(m, lv=lv+1,mode=mode)
        elif cls_name(x)=='UNetMidBlock2D':
            print_indented_with_type(lv,x,'')
            for m in x.resnets: simple_describe(m, lv=lv+1,mode=mode)
        # -- combined blocks - up
        elif cls_name(x)=='CrossAttnUpBlock2D':
            print_indented_with_type(lv,x,'')
            for m in chain.from_iterable(zip(x.resnets, x.attentions)): simple_describe(m, lv=lv+1,mode=mode)
            if hasattr(x,'upsamplers') and x.upsamplers is not None:
                for m in x.upsamplers: simple_describe(m, lv=lv+1,mode=mode)
        elif cls_name(x)=='UpBlock2D':
            print_indented_with_type(lv,x,'')
            for m in x.resnets: simple_describe(m, lv=lv+1,mode=mode)
            if hasattr(x,'upsamplers') and x.upsamplers is not None:
                for m in x.upsamplers: simple_describe(m, lv=lv+1,mode=mode)
        elif cls_name(x)=='Upsample2D': print_indented_with_type(lv, x, (x.conv.in_channels, x.conv.out_channels))
        # -- lists
        elif cls_name(x) in ['ModuleList','list','tuple','EmbedSequential']: # EmbedSequential is custom class
            print_indented_with_type(lv,x,'')
            for m in x: simple_describe(m, lv=lv+1,mode=mode)
        # -- everything else
        else: print_indented_with_type(lv,x,'')

    else: raise NotImplementedError()


def gether_channel_sizes(m, m_type):
    if m_type == 'base':
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}
        # 3.1 - input convolution
        ch_inout_base['enc'].append((m.conv_in.in_channels, m.conv_in.out_channels))
        # 3.2 - encoder blocks
        for module in m.down_blocks:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                for r in module.resnets:
                    ch_inout_base['enc'].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    ch_inout_base['enc'].append((module.downsamplers[0].channels, module.downsamplers[0].out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')
        # 3.3 - middle block
        ch_inout_base['mid'].append((m.mid_block.resnets[0].in_channels, m.mid_block.resnets[0].out_channels))
        # 3.4 - decoder blocks
        for module in m.up_blocks:
            if isinstance(module, (CrossAttnUpBlock2D, UpBlock2D)):
                for r in module.resnets:
                    ch_inout_base['dec'].append((r.in_channels, r.out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')
        return ch_inout_base
    elif m_type == 'control':
        ch_inout_ctrl = {'enc': [], 'mid': [], 'dec': []}
        # 3.1 - input convolution
        ch_inout_ctrl['enc'].append((m.conv_in.in_channels, m.conv_in.out_channels))
        # 3.2 - encoder blocks
        for module in m.down_blocks:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                for r in module.resnets:
                    ch_inout_ctrl['enc'].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    ch_inout_ctrl['enc'].append((module.downsamplers[0].channels, module.downsamplers[0].out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')
        # 3.3 - middle block
        ch_inout_ctrl['mid'].append((m.mid_block.resnets[0].in_channels, m.mid_block.resnets[0].out_channels))
        return ch_inout_ctrl
    else: raise ValueError(f'model_type must be `base` or `control`, not `{m_type}`')

def print_channels(ch_szs):
    for k,v in ch_szs.items(): print(k,v)


import re
def get_state_dict(sdict, contains='', lv=None):
    if isinstance(sdict, list): sdict = {o:'' for o in sdict} # to make it work for lists
    if lv is None: keys = [k for k in sdict.keys()]
    else: keys = {'.'.join(k.split('.')[:lv+1]): '' for k in sdict.keys()}.keys()
    contains = contains.replace('*', '[^.]*?').replace('.', r'\.') # * matches everything expect . ; . is literal
    pattern = re.compile(contains)
    return [k for k in keys if pattern.search(k)]

def print_state_dict(sdict, contains='', lv=None):
    keys = get_state_dict(sdict, contains,lv)
    for k in keys: print(k)

from collections import defaultdict
def to_nested_dict(l):
    root = {}
    for o in l:
        parts = o.split(".")
        d = root
        for part in parts:
            d = d.setdefault(part, {})
    return root

def pretty_print_dict(d,lv=2,indent=0,depth=1,print_leaf=False):
    if depth > lv: return
    for k,v in d.items():
        print('  ' * indent + str(k))
        if isinstance(v, dict): pretty_print_dict(v,lv,indent+4,depth+1,print_leaf=print_leaf)
        else: 
            if print_leaf: print('  ' * (indent+1) + str(v))
            
def print_as_nested_dict(l,lv=1,print_leaf=False): pretty_print_dict(to_nested_dict(l),lv=lv,print_leaf=print_leaf)
