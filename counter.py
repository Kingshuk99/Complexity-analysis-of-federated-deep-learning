#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from results import *
from collections import OrderedDict
from computations import compute_flops, compute_params, compute_fmap, compute_interm_rep


# In[2]:


class ModelSummary_f(object):
    def __init__(self, model, input_size, batch_size=1, device='cuda'):
        super(ModelSummary_f, self).__init__()
        assert device.lower() in ['cuda', 'cpu']
        self.model = model.to(device)
        self.batch_size = batch_size

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = list(input_size)
        self.input_size = input_size

        # batch_size of 2 for batchnorm
        x = torch.rand([2] + input_size).type(dtype)

        # create properties
        self.summary = OrderedDict()
        self.f_hooks = list()

        # register hook

        model.apply(self.register_hook)

        '''.apply() Applies some fn recursively to every 
         submodule (as returned by .children())
         as well as self. Typical use includes 
         initializing the parameters of a model.'''

        # make a forward pass
        out = model(x)

        # remove hooks
        for h in self.f_hooks:
            h.remove()


    def register_hook(self, module):
        if len(list(module.children())) == 0:
            ''' .children() Returns an iterator over immediate children modules.'''

            self.f_hooks.append(module.register_forward_hook(self.f_hook))

    def f_hook(self, module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(self.summary)

        m_key = "%s-%i" % (class_name, module_idx + 1)
        self.summary[m_key] = OrderedDict()
        self.summary[m_key]["input_shape"] = list(input[0].size())
        if isinstance(output, (list, tuple)):
            self.summary[m_key]["output_shape"] = [[self.batch_size] + list(o.size())[1:] for o in output]
        else:
            self.summary[m_key]["output_shape"] = [self.batch_size] + list(output.size()[1:])

        # -------------------------
        # compute module parameters
        # -------------------------
        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += int(torch.prod(torch.LongTensor(list(module.weight.size()))))
            self.summary[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += int(torch.prod(torch.LongTensor(list(module.bias.size()))))
        self.summary[m_key]["nb_params"] = params
        self.summary[m_key]["calc_params"] = compute_params(module, input[0], output)

        # -------------------------
        # compute module flops
        # -------------------------

        flops = compute_flops(module, input[0], output, batch_size = self.batch_size)
        self.summary[m_key]["f_flops"] = flops
        
        flops_inf = compute_flops(module, input[0], output, mode = 'infer')
        self.summary[m_key]["inf_flops"] = flops_inf
        
        layer = module_idx + 1
        fmap = compute_fmap(module, input[0], output, l = layer, batch_size = self.batch_size)
        self.summary[m_key]["fmap"] = fmap
        self.summary[m_key]["exp_fmap"] = output.element_size()*output.nelement()
        int_rep = compute_interm_rep(module, input[0], output, l = layer, batch_size = self.batch_size)
        self.summary[m_key]["interm_rep"] = int_rep
#         if len(list(self.summary.items()))==1:
#             self.summary[m_key]["exp_fmap"] = torch.cuda.memory_allocated()
#         else:
#             self.summary[m_key]["exp_fmap"] = torch.cuda.memory_allocated()-self.summary[list(self.summary.items())[-2][0]]["exp_fmap"]
        

    def show_f(self):
        print("-----------------------------------------Forward Pass - Training------------------------------------------------------")
        line = "{:>25}  {:>15} {:>10} {:>15} {:>15} {:>10} {:>10} {:>10}".format("Layer (type)", "Output Shape", "Params", "Addition", 'Multiplication', 'Logical', 'Sq inv', 'Division')
        print(line)
        print("======================================================================================================================")
        total_params, total_output, trainable_params, total_flops = 0, 0, 0, 0
        for layer in self.summary:
            if(self.summary[layer]["nb_params"] != self.summary[layer]["calc_params"]):
                print("Calculation of no. of parameters of {} is not correct, it has {} parameters, calculated value is {}".format(layer, self.summary[layer]["nb_params"], self.summary[layer]["calc_params"]))
                return 0
            line = "{:>25}  {:>15} {:>10} {:>15} {:>15} {:>10} {:>10} {:>10}".format(
                layer,
                str(self.summary[layer]["output_shape"]),
                "{0:,}".format(self.summary[layer]["nb_params"]),
                "{0:,}".format(self.summary[layer]["f_flops"]['add']),
                "{0:,}".format(self.summary[layer]["f_flops"]['mul']),
                "{0:,}".format(self.summary[layer]["f_flops"]['log']),
                "{0:,}".format(self.summary[layer]["f_flops"]['sq_inv']),
                "{0:,}".format(self.summary[layer]["f_flops"]['div'])
            )
            total_params += self.summary[layer]["nb_params"]
            total_output += np.prod(self.summary[layer]["output_shape"])
            total_flops += (self.summary[layer]["f_flops"]['add']+self.summary[layer]["f_flops"]['mul']+self.summary[layer]["f_flops"]['log']+self.summary[layer]["f_flops"]['sq_inv']+self.summary[layer]["f_flops"]['div'])

            if "trainable" in self.summary[layer]:
                if self.summary[layer]["trainable"] == True:
                    trainable_params += self.summary[layer]["nb_params"]
            print(line)

        total_input_size = abs(np.prod(self.input_size) * self.batch_size / (1024 ** 2.))
        total_output_size = abs(2. * total_output / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params / (1024 ** 2.))
        total_flops_size = abs(total_flops / (1e9))
        total_size = total_params_size + total_output_size + total_input_size
        model_name = str(self.model.__class__).split(' ')[-1].split('.')[-1].split("'")[0]
        df1 = show_forward_training(model_name, self.summary)

        print("======================================================================================================================")
        print("Total params: {0:,}".format(total_params))
        #print("Trainable params: {0:,}".format(trainable_params))
        #print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("Total FLOPs: {0:,}".format(total_flops))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("FLOPs size (GB): %0.2f" % total_flops_size)
        print("----------------------------------------------------------------")
        print("-----------------------------------------Forward Pass - Inference-----------------------------------------------------")
        line = "{:>25}  {:>15} {:>15} {:>15} {:>10} {:>10} {:>10}".format("Layer (type)", "Output Shape", "Addition", 'Multiplication', 'Logical', 'Sq inv', 'Division')
        print(line)
        print("======================================================================================================================")
        total_params, total_output, trainable_params, total_flops = 0, 0, 0, 0
        for layer in self.summary:
            line = "{:>25}  {:>15} {:>15} {:>15} {:>10} {:>10} {:>10}".format(
                layer,
                str(self.summary[layer]["output_shape"]),
                "{0:,}".format(self.summary[layer]["inf_flops"]['add']),
                "{0:,}".format(self.summary[layer]["inf_flops"]['mul']),
                "{0:,}".format(self.summary[layer]["inf_flops"]['log']),
                "{0:,}".format(self.summary[layer]["inf_flops"]['sq_inv']),
                "{0:,}".format(self.summary[layer]["inf_flops"]['div'])
            )
            total_output += np.prod(self.summary[layer]["output_shape"])
            total_flops += (self.summary[layer]["inf_flops"]['add']+self.summary[layer]["inf_flops"]['mul']+self.summary[layer]["inf_flops"]['log']+self.summary[layer]["inf_flops"]['sq_inv']+self.summary[layer]["inf_flops"]['div'])

            print(line)

        total_input_size = abs(np.prod(self.input_size) * self.batch_size / (1024 ** 2.))
        total_output_size = abs(2. * total_output / (1024 ** 2.))  # x2 for gradients
        total_flops_size = abs(total_flops / (1e9))
        df2 = show_forward_inference(model_name, self.summary)

        print("======================================================================================================================")
        print("Total FLOPs: {0:,}".format(total_flops))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward pass size (MB): %0.2f" % total_output_size)
        print("FLOPs size (GB): %0.2f" % total_flops_size)
        print("----------------------------------------------------------------")
        return 1, len(self.summary), df1, df2


# In[ ]:


class ModelSummary_b(object):
    def __init__(self, model, input_size, length, batch_size=1, criterion = nn.CrossEntropyLoss(), device='cuda'):
        super(ModelSummary_b, self).__init__()
        assert device.lower() in ['cuda', 'cpu']
        self.model = model.to(device)
        self.batch_size = batch_size
        self.length = length

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = list(input_size)
        self.input_size = input_size

        # batch_size of 2 for batchnorm
        x = torch.rand([2] + input_size).type(dtype)
        x.requires_grad = True

        # create properties
        self.summary = OrderedDict()
        self.b_hooks = list()

        # register hook

        model.apply(self.register_hook)

        '''.apply() Applies some fn recursively to every 
         submodule (as returned by .children())
         as well as self. Typical use includes 
         initializing the parameters of a model.'''

        # make a forward pass
        out = model(x)
        n = out.size()[-1]
        target = torch.randint(0, n, (2,)).to(device)
        err = criterion(out, target).to(device)
        err.backward()

        # remove hooks
            
        for h in self.b_hooks:
            h.remove()

    def register_hook(self, module):
        if len(list(module.children())) == 0:
            ''' .children() Returns an iterator over immediate children modules.'''

#             self.b_hooks.append(module.register_backward_hook(self.b_hook))
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if class_name == 'Conv2d' or class_name == 'Linear' or (hasattr(module, "inplace") and module.inplace==False):
                self.b_hooks.append(module.register_full_backward_hook(self.b_hook))
            else:
                self.b_hooks.append(module.register_backward_hook(self.b_hook))

    def b_hook(self, module, grad_in, grad_out):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(self.summary)

        m_key = "%s-%i" % (class_name, self.length - module_idx)
        self.summary[m_key] = OrderedDict()
        self.summary[m_key]["input_shape"] = list(grad_in[0].size())
        if isinstance(grad_out, (list, tuple)):
            self.summary[m_key]["output_shape"] = [[self.batch_size] + list(o.size())[1:] for o in grad_out][0]
        else:
            self.summary[m_key]["output_shape"] = [self.batch_size] + list(grad_out.size())

        # -------------------------
        # compute module parameters
        # -------------------------
        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += int(torch.prod(torch.LongTensor(list(module.weight.size()))))
            self.summary[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += int(torch.prod(torch.LongTensor(list(module.bias.size()))))
        self.summary[m_key]["nb_params"] = params
        self.summary[m_key]["calc_params"] = compute_params(module, grad_in[0], grad_out)

        # -------------------------
        # compute module flops
        # -------------------------

        flops = compute_flops(module, grad_in[0], grad_out, batch_size = self.batch_size, p = 'b')
        self.summary[m_key]["b_flops"] = flops
        
        layer = self.length - module_idx
        
        fmap = compute_fmap(module, grad_in[0], grad_out, l = layer, batch_size = self.batch_size, p = 'b')
        self.summary[m_key]["fmap"] = fmap
        int_rep = compute_interm_rep(module, grad_in[0], grad_out, l = layer, batch_size = self.batch_size, p = 'b')
        self.summary[m_key]["interm_rep_grad"] = int_rep
        if layer==1 and class_name == "Conv2d":
            self.summary[m_key]["exp_fmap"] = 0
        else:
            self.summary[m_key]["exp_fmap"] = grad_in[0].element_size()*grad_in[0].nelement()

#         if len(list(self.summary.items()))==1:
#             self.summary[m_key]["exp_fmap"] = torch.cuda.memory_allocated()
#         else:
#             self.summary[m_key]["exp_fmap"] = torch.cuda.memory_allocated()-self.summary[list(self.summary.items())[-2][0]]["exp_fmap"]
        

    def show_b(self):
        print("-----------------------------------------------Backward Pass----------------------------------------------------------")
        line = "{:>25}  {:>15} {:>15} {:>15} {:>10} {:>10} {:>10}".format("Layer (type)", "Output Shape", "Addition", 'Multiplication', 'Logical', 'Sq inv', 'Division')
        print(line)
        print("======================================================================================================================")
        total_output, total_flops = 0, 0
        for layer in self.summary:
            line = "{:>25}  {:>15} {:>15} {:>15} {:>10} {:>10} {:>10}".format(
                layer,
                str(self.summary[layer]["output_shape"]),
                "{0:,}".format(self.summary[layer]["b_flops"]['add']),
                "{0:,}".format(self.summary[layer]["b_flops"]['mul']),
                "{0:,}".format(self.summary[layer]["b_flops"]['log']),
                "{0:,}".format(self.summary[layer]["b_flops"]['sq_inv']),
                "{0:,}".format(self.summary[layer]["b_flops"]['div'])
            )
            #total_params += self.summary[layer]["nb_params"]
            total_output += np.prod(self.summary[layer]["output_shape"])
            total_flops += (self.summary[layer]["b_flops"]['add']+self.summary[layer]["b_flops"]['mul']+self.summary[layer]["b_flops"]['log']+self.summary[layer]["b_flops"]['sq_inv']+self.summary[layer]["b_flops"]['div'])

#             if "trainable" in self.summary[layer]:
#                 if self.summary[layer]["trainable"] == True:
#                     trainable_params += self.summary[layer]["nb_params"]
            print(line)

        total_input_size = abs(np.prod(self.input_size) * self.batch_size / (1024 ** 2.))
        total_output_size = abs(2. * total_output / (1024 ** 2.))  # x2 for gradients
        #total_params_size = abs(total_params.numpy() / (1024 ** 2.))
        total_flops_size = abs(total_flops / (1e9))
        #total_size = total_params_size + total_output_size + total_input_size
        model_name = str(self.model.__class__).split(' ')[-1].split('.')[-1].split("'")[0]
        df3 = show_backward(model_name, self.summary)

        print("======================================================================================================")
        #print("Total params: {0:,}".format(total_params))
        #print("Trainable params: {0:,}".format(trainable_params))
        #print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("Total FLOPs: {0:,}".format(total_flops))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Backward pass size (MB): %0.2f" % total_output_size)
        #print("Params size (MB): %0.2f" % total_params_size)
        #print("Estimated Total Size (MB): %0.2f" % total_size)
        print("FLOPs size (GB): %0.2f" % total_flops_size)
        print("----------------------------------------------------------------")
        return df3


# In[ ]:


def scope(model, input_size, batch_size=1, criterion = nn.CrossEntropyLoss(), device='cuda'):
    summary_f = ModelSummary_f(model, input_size, batch_size, device)
    temp, length, df1, df2 = summary_f.show_f()
    if(temp==1):
        summary_b = ModelSummary_b(model, input_size,length, batch_size, criterion, device)
        df3 = summary_b.show_b()
    model_name = str(model.__class__).split(' ')[-1].split('.')[-1].split("'")[0]
    with pd.ExcelWriter(model_name+'.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Forward pass for training')
        df2.to_excel(writer, sheet_name='Forward pass for inference')
        df3.to_excel(writer, sheet_name='Backward pass')

