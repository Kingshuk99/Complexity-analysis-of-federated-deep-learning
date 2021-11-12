#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


# In[2]:


__all__ = ["compute_flops"]


# In[ ]:


def compute_flops(module, inp, out, batch_size = 1, p = 'f', mode = 'train'):
    res = OrderedDict()
    res['add'] = 0
    res['mul'] = 0
    res['log'] = 0
    res['div'] = 0
    res['sq_inv'] = 0
    res['exp'] = 0
    res['logar'] = 0
    res['inv'] = 0
    if isinstance(module, nn.Conv2d):
        add, mul = compute_Conv2d_flops(module, inp, out, passing = p)
        res['add'] = add*batch_size//2
        res['mul'] = mul*batch_size//2
        return res
    elif isinstance(module, nn.BatchNorm2d):
        if p == 'f':
            if mode == 'train':
                add, mul, sq_inv = compute_BatchNorm2d_flops(module, inp, out, b_sz = batch_size)
            else:
                add, mul, sq_inv = compute_BatchNorm2d_flops(module, inp, out, b_sz = batch_size, mode = 'infer')
        else:
            add, mul, sq_inv = compute_BatchNorm2d_flops(module, inp, out, b_sz = batch_size, passing = 'b')
        res['add'] = add
        res['mul'] = mul
        res['sq_inv'] = sq_inv
        return res
    elif isinstance(module,  nn.MaxPool2d):
        if p == 'f':
            res['log'] = compute_Pool2d_flops(module, inp, out)*batch_size//2
        else:
            res['add'] = compute_Pool2d_flops(module, inp, out, passing = 'b')*batch_size//2
        return res
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
        res['log'] = compute_ReLU_flops(module, inp, out)*batch_size//2
        return res
    elif isinstance(module,  nn.AdaptiveAvgPool2d):
        add, div = compute_AvgPool2d_flops(module, inp, out, passing = p)
        res['add'] = add*batch_size
        res['div'] = div*batch_size
        return res
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU)):
        res['log'] = compute_ReLU_flops(module, inp, out, passing = p)*batch_size//2
        return res
    elif isinstance(module, nn.PReLU):
        log, mul, add = compute_PReLU_flops(module, inp, out, passing = p)*batch_size//2
        res['log'] = log
        res['mul'] = mul
        res['add'] = add
        return res
    elif isinstance(module, nn.Tanh):
        exp, mul, add, div = compute_PReLU_flops(module, inp, out, passing = p)*batch_size//2
        res['exp'] = exp
        res['mul'] = mul
        res['add'] = add
        res['div'] = div
        return res
    elif isinstance(module, nn.Softplus):
        exp, logar, add, div = compute_Softplus_flops(module, _inp, _out, passing = p)*batch_size//2
        res['exp'] = exp
        res['logar'] = logar
        res['add'] = add
        res['div'] = div
        return res
    elif isinstance(module, nn.Sigmoid):
        exp, mul, add, div = compute_Sigmoid_flops(module, _inp, _out, passing = p)*batch_size//2
        res['exp'] = exp
        res['mul'] = mul
        res['add'] = add
        res['div'] = div
        return res
    elif isinstance(module, nn.Softmax):
        exp, mul, add, inv = compute_Sigmoid_flops(module, _inp, _out, passing = p)*batch_size//2
        res['exp'] = exp
        res['mul'] = mul
        res['add'] = add
        res['inv'] = inv
        return res
#     elif isinstance(module, nn.Upsample):
#         return compute_Upsample_flops(module, inp, out) // 2
    elif isinstance(module, nn.Linear):
        add, mul = compute_Linear_flops(module, inp, out, passing = p)
        res['add'] = add*batch_size//2
        res['mul'] = mul*batch_size//2
        return res
    elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
        res['mul'] = compute_Dropout_flops(module, inp, out)
        return res
    else:
        return res


# In[ ]:


def compute_params(module, inp, out, p = 'f'):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_params(module, inp, out, passing = p)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_params(module, inp, out, passing = p)
    elif isinstance(module, nn.Linear):
        return compute_Linear_params(module, inp, out, passing = p)
    else:
        return 0


# In[ ]:


def compute_fmap(module, inp, out, l, batch_size = 1, p = 'f'):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_fmap(module, inp, out, layer = l, passing = p)*batch_size
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_fmap(module, inp, out, passing = p)*batch_size
    elif isinstance(module,  (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        return compute_Pool2d_fmap(module, inp, out, passing = p)*batch_size
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Softplus, nn.Sigmoid, nn.Softmax)):
        return compute_ReLU_fmap(module, inp, out)*batch_size
#     elif isinstance(module, nn.Upsample):
#         return compute_Upsample_flops(module, inp, out) // 2
    elif isinstance(module, nn.Linear):
        return compute_Linear_fmap(module, inp, out, passing = p)*batch_size
    else:
        return 0


# In[ ]:


def compute_interm_rep(module, inp, out, l, batch_size = 1, p = 'f'):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_interm_rep(module, inp, out, layer = l, passing = p)*batch_size
    else:
        return 0


# In[3]:


def compute_Conv2d_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+list( _out[0].size()[0:]))
    else:
        out = _out[0].size()
    assert isinstance(module, nn.Conv2d)
    #assert len(inp) == 4 and len(inp) == len(out)
    if isinstance(module.kernel_size, (tuple, list)):
        kh, kw = module.kernel_size
    else:
        kh, kw = module.kernel_size, module.kernel_size
    if passing == 'f':
        if module.bias is not None:
            add = out[3]*out[3]*kh*kw*inp[1]*out[1]
            mul = add
        else:
            add = out[2]*out[3]*(kh*kw*inp[1]-1)*out[1]
            mul = out[2]*out[3]*kh*kw*inp[1]*out[1]
    else:
        add = (out[2]*out[3]-1)*kh*kw*inp[1]*out[1]+out[2]*out[3]*(out[1]-1)*kh*kw*inp[1]+(out[2]*out[3]*kh*kw-inp[2]*inp[3])*inp[1]
        mul = 2*out[2]*out[3]*kh*kw*inp[1]*out[1]
        if module.bias is not None:
            add += (out[2]*out[3]-1)*out[1]
    return int(add*inp[0]), int(mul*inp[0])


# In[4]:


def compute_Pool2d_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+list( _out[0].size()[0:]))
    else:
        out = _out[0].size()
    assert isinstance(module, nn.MaxPool2d)
    #assert len(inp) == 4 and len(inp) == len(out)
    if isinstance(module.kernel_size, (tuple, list)):
        kh, kw = module.kernel_size
    else:
        kh, kw = module.kernel_size, module.kernel_size
    if passing == 'f':
        op = out[1]*out[2]*out[3]*(kh*kw-1)
    else :
        op = out[1]*out[2]*out[3]
    return int(op*inp[0])


# In[ ]:


def compute_AvgPool2d_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+list( _out[0].size()[0:]))
    else:
        out = _out[0].size()
    assert isinstance(module, nn.AdaptiveAvgPool2d)
    if passing == 'f':
        add = out[1]*(inp[2]*inp[3]-1)
        div = out[1]
    else:
        add = out[1]*inp[2]*inp[3]
        div = out[1]
    return int(add), int(div)


# In[5]:


def compute_Linear_flops(module, _inp, _out, passing = 'f'):
    if len(_out)>1:
        out = torch.tensor([len(_out), _out[0].size()[-1]])
    else:
        out = _out[0].size()
    assert isinstance(module, nn.Linear)
    #assert len(inp) == 2 and len(out) == 2
    if passing == 'f':
        inp = _inp.size()
        add = out[1]*inp[-1]*out[0]
        mul = add
    else:
        inp = module.weight.size()[-1]
        add = inp*(out[1]-1)*out[0]
        mul = 2*out[1]*inp*out[0]
    return int(add), int(mul)


# In[10]:


def compute_ReLU_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+ list(_out[0].size()))
    else:
        out = _out[0].size()
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU))
    ans = int(torch.prod(torch.LongTensor(list(out))))
    return ans


# In[ ]:


def compute_PReLU_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+ list(_out[0].size()))
    else:
        out = _out[0].size()
    assert isinstance(module, nn.PReLU)
    if passing == 'f':
        log = int(torch.prod(torch.LongTensor(list(out))))
        mul = log
        add = 0
    else:
        log = 2*int(torch.prod(torch.LongTensor(list(out))))
        mul = 1
        add = int(torch.prod(torch.LongTensor(list(out))))-1
    return log, mul, add


# In[ ]:


def compute_Tanh_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+ list(_out[0].size()))
    else:
        out = _out[0].size()
        assert isinstance(module, nn.Tanh)
    if passing == 'f':
        exp = int(torch.prod(torch.LongTensor(list(out))))
        mul = int(torch.prod(torch.LongTensor(list(out))))
        add = 2*int(torch.prod(torch.LongTensor(list(out))))
        div = int(torch.prod(torch.LongTensor(list(out))))
    else:
        exp = 0
        mul = int(torch.prod(torch.LongTensor(list(out))))*2
        add = int(torch.prod(torch.LongTensor(list(out))))
        div = 0
    return exp, mul, add, div


# In[ ]:


def compute_Sigmoid_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+ list(_out[0].size()))
    else:
        out = _out[0].size()
        assert isinstance(module, nn.Sigmoid)
    if passing == 'f':
        exp = int(torch.prod(torch.LongTensor(list(out))))
        mul = 0
        add = int(torch.prod(torch.LongTensor(list(out))))
        div = int(torch.prod(torch.LongTensor(list(out))))
    else:
        exp = 0
        mul = int(torch.prod(torch.LongTensor(list(out))))*2
        add = int(torch.prod(torch.LongTensor(list(out))))
        div = 0
    return exp, mul, add, div


# In[ ]:


def compute_Softplus_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+ list(_out[0].size()))
    else:
        out = _out[0].size()
        assert isinstance(module, nn.Softplus)
    if passing == 'f':
        exp = int(torch.prod(torch.LongTensor(list(out))))
        logar = int(torch.prod(torch.LongTensor(list(out))))
        add = int(torch.prod(torch.LongTensor(list(out))))
        div = 0
    else:
        exp = 0
        logar = 0
        add = 0
        div = int(torch.prod(torch.LongTensor(list(out))))
    return exp, logar, add, div


# In[ ]:


def compute_Softmax_flops(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+ list(_out[0].size()))
    else:
        out = _out[0].size()
        assert isinstance(module, nn.Softmax)
    if passing == 'f':
        exp = out[1]
        mul = out[1]
        add = out[1]-1
        inv = 1
    else:
        exp = 0
        mul = out[1]-1
        add = out[1]*2-1
        inv = 0
    return int(exp), int(mul), int(add), int(inv)


# In[ ]:


def compute_BatchNorm2d_flops(module, _inp, _out, b_sz, passing = 'f', mode = 'train'):
    inp = _inp.size()
    if b_sz==1:
        b_sz = 2
    if passing == 'f':
        if mode == 'train':
            add = (2*inp[2]*inp[3]*b_sz+1)*inp[1]
            mul = (3*inp[2]*inp[3]*b_sz+6)*inp[1]
            sq_inv = inp[1]
        else:
            add = (2*inp[2]*inp[3]+1)*inp[1]*b_sz
            mul = (inp[2]*inp[3]+1)*inp[1]*b_sz
            sq_inv = inp[1]*b_sz
    else :
        add = (8*inp[2]*inp[3]*b_sz-4)*inp[1]
        mul = (7*inp[2]*inp[3]*b_sz+8)*inp[1]
        sq_inv = 0
    return int(add), int(mul), int(sq_inv)


# In[ ]:


def compute_Dropout_flops(module, _inp, _out):
    inp = _inp.size()
    return int(torch.prod(torch.LongTensor(list(inp))))


# In[11]:


def compute_Conv2d_params(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+list( _out[0].size()[0:]))
    else:
        out = _out[0].size()
    if isinstance(module.kernel_size, (tuple, list)):
        kh, kw = module.kernel_size
    else:
        kh, kw = module.kernel_size, module.kernel_size
    param = kh*kw*inp[1]*out[1]
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        param += out[1]
    return int(param)


# In[12]:


def compute_Linear_params(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out), _out[0].size()[-1]])
    else:
        out = _out[0].size()
    assert isinstance(module, nn.Linear)
    param = inp[1]*out[1]
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        param += out[1]
    return int(param)


# In[ ]:


def compute_BatchNorm2d_params(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    return int(2*inp[1])


# In[ ]:


def compute_Conv2d_fmap(module, _inp, _out, layer = 0, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+list( _out[0].size()[0:]))
    else:
        out = _out[0].size()
    assert isinstance(module, nn.Conv2d)
    #assert len(inp) == 4 and len(inp) == len(out)
    if isinstance(module.kernel_size, (tuple, list)):
        kh, kw = module.kernel_size
    else:
        kh, kw = module.kernel_size, module.kernel_size
#     if passing=='b':
#         print(inp)
#         print(out)
    if layer == 1:
        if passing == 'f':
            fmap = out[1]*out[2]*out[3]
        else:
            fmap = 0
    else:
        if passing == 'f':
            fmap = out[1]*out[2]*out[3]
        else:
            fmap = inp[1]*inp[2]*inp[3]
#             print(kh)
#             print(kw)
#             print(inp[1])
#             print(inp[2])
#             print(inp[3])
#             print(out[2])
#             print(out[3])
#             print(fmap)
    return int(fmap)


# In[ ]:


def compute_Conv2d_interm_rep(module, _inp, _out, layer = 0, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+list( _out[0].size()[0:]))
    else:
        out = _out[0].size()
    assert isinstance(module, nn.Conv2d)
    #assert len(inp) == 4 and len(inp) == len(out)
    if isinstance(module.kernel_size, (tuple, list)):
        kh, kw = module.kernel_size
    else:
        kh, kw = module.kernel_size, module.kernel_size
    if layer == 1:
        if passing == 'f':
            int_rep = out[2]*out[3]*inp[1]*kh*kw
        else:
            int_rep = 0
    else:
        if passing == 'f':
            int_rep = out[2]*out[3]*inp[1]*kh*kw
        else:
            int_rep = out[2]*out[3]*kh*kw*inp[1]
    return int(int_rep)


# In[14]:


def compute_Pool2d_fmap(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+list( _out[0].size()[0:]))
    else:
        out = _out[0].size()
    assert isinstance(module, (nn.MaxPool2d, nn.AdaptiveAvgPool2d))
    if passing == 'f':
        return int(out[1]*out[2]*out[3])
    else:
        return int(inp[1]*inp[2]*inp[3])


# In[ ]:


def compute_Linear_fmap(module, _inp, _out, passing = 'f'):
    if len(_out)>1:
        out = torch.tensor([len(_out), _out[0].size()[-1]])
    else:
        out = _out[0].size()
    assert isinstance(module, nn.Linear)
    #assert len(inp) == 2 and len(out) == 2
    if passing == 'f':
        inp = _inp.size()
        return int(out[1])
    else:
        inp = module.weight.size()[-1]
        return int(inp)


# In[ ]:


def compute_ReLU_fmap(module, _inp, _out):
    inp = _inp.size()
    if len(_out)>1:
        out = torch.tensor([len(_out)]+ list(_out[0].size()))
    else:
        out = _out[0].size()
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU, nn.Tanh, nn.Softplus, nn.Sigmoid, nn.Softmax))
    return int(torch.prod(torch.LongTensor(list(out))))/inp[0]


# In[ ]:


def compute_BatchNorm2d_fmap(module, _inp, _out, passing = 'f'):
    inp = _inp.size()
    if passing == 'f':
        return 2*int(torch.prod(torch.LongTensor(list(inp))))
    else :
        return 3*int(torch.prod(torch.LongTensor(list(inp))))

