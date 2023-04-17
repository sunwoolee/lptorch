import torch
import torch.nn as nn
import onnx

class Quantized_Model(torch.nn.Module):
    def __init__(self, model):
        super(Quantized_Model, self).__init__()
        self.lp_module = model
        self.bn_merge = False
        self.merged_bn = []
        self.base_parameters = {}        # to be quantized
        self.bias_parameters = {}
        self.other_parameters = {}  # not to be quantized
        for p in model.named_parameters():
            self.base_parameters[p[0]] = p[1]
        bn_parameters = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_parameters += list(m.parameters())
        # remove bn parameters
        bn_keys = []
        for k,v in self.base_parameters.items():
            match = False
            for bn in bn_parameters:
                if bn is v:
                    match = True
                    break
            if match:
                bn_keys.append(k)
        for k in bn_keys:
            self.other_parameters[k] = self.base_parameters[k]
            del self.base_parameters[k]
        # remove bias parameters
        bias_keys = []
        for k in self.base_parameters.keys():
            if 'bias' in k:
                bias_keys.append(k)
        for k in bias_keys:
            self.bias_parameters[k] = self.base_parameters[k]
            del self.base_parameters[k]
        
        # make master copy & previous quant value
        self.master = {}
        self.previous = {}
        for k,v in self.base_parameters.items():
            self.master[k] = v.data.clone()
            self.previous[k] = v.data.clone()
        self.bias_master = {}
        self.bias_previous = {}
        for k,v in self.bias_parameters.items():
            self.bias_master[k] = v.data.clone()
            self.bias_previous[k] = v.data.clone()

    def forward(self, *args, **kwargs):
        return self.lp_module(*args, **kwargs)

    def train(self, mode = True):
        self.lp_module.train(mode)
        for bn in self.merged_bn:
            bn.eval()

    def eval(self):
        self.lp_module.eval()

    def parameter_quantize(self):
        from .major import weight_quant, bias_quant, hysteresis_update
        for k,v in self.base_parameters.items():
            self.master[k].data = v.clone().data
        for k,v in self.bias_parameters.items():
            self.bias_master[k].data = v.clone().data
        if weight_quant is not None:
            for k,v in self.master.items():
                if hysteresis_update:
                    self.base_parameters[k].data = weight_quant.hysteresis(self.previous[k].data, v.clone()).data
                else:
                    self.base_parameters[k].data = weight_quant.quantize(v.clone()).data
        if bias_quant is not None:
            for k,v in self.bias_master.items():
                if hysteresis_update:
                    self.bias_parameters[k].data = bias_quant.hysteresis(self.bias_previous[k].data, v.clone()).data
                else:
                    self.bias_parameters[k].data = bias_quant.quantize(v.clone()).data

    def parameter_recover(self):
        for k,v in self.master.items():
            self.previous[k].data = self.base_parameters[k].clone().data
            self.base_parameters[k].data = v.clone().to(self.base_parameters[k].device).data
        for k,v in self.bias_master.items():
            self.bias_previous[k].data = self.bias_parameters[k].clone().data
            self.bias_parameters[k].data = v.clone().to(self.bias_parameters[k].device).data

    def merge_bn(self, original_model, dummy_input, scale_only=False):
        torch.onnx.export(original_model, dummy_input, 'model.onnx', verbose=True, training=torch.onnx.TrainingMode.TRAINING)
        onnx_model = onnx.load("model.onnx")

        param_dict = {}
        for n, p in self.lp_module.named_parameters():
            n = n.replace('lp_module.', '')
            param_dict[n] = p
        module_list = [nn.Conv2d, nn.Conv1d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d]

        module_dict = {}
        for m in self.lp_module.modules():
            match = False
            for module in module_list:
                if isinstance(m, module):
                    match = True
                    break
            if match:
                for mp in m.parameters():
                    for n, p in param_dict.items():
                        if mp is p:
                            module_dict[n] = m
                            break

        class node():
            def __init__(self, inputs, outputs, module):
                self.inputs = inputs
                self.outputs = outputs
                self.module = module

        Gemm_node_list = ['Conv', 'Gemm']
        BatchNorm_node_list = ['BatchNormalization']
        Gemm_node = []
        BatchNorm_node = []
        other_inputs = []
        for n in onnx_model.graph.node:
            if n.op_type in Gemm_node_list:
                module = None
                for input in n.input:
                    if input in module_dict.keys():
                        module = module_dict[input]
                        break
                if module == None:
                    raise Exception("GEMM module is not found! Check \'module_list\' and \'Gemm_node_list\'.")
                Gemm_node.append(node(n.input, n.output, module))
            elif n.op_type in BatchNorm_node_list:
                module = None
                for input in n.input:
                    if input in module_dict.keys():
                        module = module_dict[input]
                        break
                if module == None:
                    raise Exception("BatchNorm module is not found! Check \'module_list\' and 'BatchNorm_node_list\'.")
                BatchNorm_node.append(node(n.input, n.output, module))
            else:
                other_inputs += n.input
        self.merged_bn = []
        ConvBn = []
        for gemm in Gemm_node:
            count = 0
            match_BN = None
            name = gemm.outputs[0]
            for bn in BatchNorm_node:
                if name in bn.inputs:
                    count += 1
                    match_BN = bn.module
            if name in other_inputs:
                count += 2
            if count == 1:
                ConvBn.append([gemm.module, match_BN])
            if match_BN is not None:
                self.merged_bn.append(match_BN)

        for conv, bn in ConvBn:
            mul_val = bn.weight.div(torch.sqrt(bn.running_var.add(bn.eps)))
            add_val = bn.bias.add(-bn.running_mean.mul(mul_val))
            if scale_only:
                tmp = torch.pow(2,mul_val.log2().floor())
                tmp = torch.where(mul_val == 0, mul_val.add(1), tmp)
                bn_weight = mul_val.div(tmp)
                mul_val = tmp
                bn_bias = add_val
                add_val = add_val.mul(0)
            else:
                bn_weight = mul_val.mul(0).add(1)
                bn_bias = add_val.mul(0)
            shape = [-1] + [1] * (conv.weight.dim()-1)
            conv.weight.data = conv.weight.data.mul(mul_val.reshape(shape)).data
            if conv.bias is None:
                conv.bias = torch.nn.Parameter(add_val)
            else:
                conv.bias.data = conv.bias.data.mul(mul_val).add(add_val).data
            bn.weight.data = bn_weight
            bn.bias.data = bn_bias
            bn.running_mean.data = bn.running_mean.data.mul(0)
            bn.running_var.data = bn.running_var.data.mul(0).add(1)
            bn.weight.requires_grad = False
            bn.bias.requires_grad = False
            bn.eps = 0
            bn.eval()
        
        self.bias_parameters = {}
        self.bias_master = {}
        self.bias_previous = {}
        for p in self.lp_module.named_parameters():
            if 'bias' in p[0]:
                self.bias_parameters[p[0]] = p[1]
        for k,v in self.bias_parameters.items():
            self.bias_master[k] = v.data.clone()
            self.bias_previous[k] = v.data.clone()

        self.bn_merge = True
