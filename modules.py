from os import error
from numpy.lib.twodim_base import mask_indices
import torch
from . import functions as F
from . import quant as quant
from torch.nn.utils.rnn import PackedSequence
from .major import get_distribution
import pdb

class debug_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, do_pdb):
        ctx.do_pdb = do_pdb
        return input
    @staticmethod
    def backward(ctx, grad_output):
        do_pdb = ctx.do_pdb
        if do_pdb:
            pdb.set_trace()
        torch.save(grad_output, 'grad.pt')
        return grad_output, None

class Debug(torch.nn.Module):
    def __init__(self, do_pdb=False):
        super().__init__()
        self.do_pdb = do_pdb

    def forward(self, input):
        output = debug_func.apply(input, self.do_pdb)
        output.scale = input.scale
        return output
    
class qblock_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, tracked_scale, dual, training, tracking, quantize, shortcut, distribution, quant_type):
        ctx.scale = scale
        ctx.tracked_scale = tracked_scale
        ctx.dual = dual
        ctx.tracking = tracking
        ctx.quantize = quantize
        ctx.shortcut = shortcut
        ctx.distribution = distribution
        from .major import track_distribution
        if track_distribution:
            get_distribution(input, distribution['forward'])
        if shortcut:
            from .major import shortcut_quant as activ_quant
        else:
            from .major import activ_quant
        if activ_quant is None or quantize[0] is False:\
            return input
        if not training:
            scale[0] = scale[0].clone()
            tracked_scale[0] = tracked_scale[0].clone()
        room = None
        forward_scale = scale[0].clone()
        if tracking[0]:
            if dual[0]:
                output = activ_quant.quantize(input.clone(), scale[0], room, quant_type=quant_type)
                output.add_(activ_quant.quantize(input.add(-output), forward_scale, quant_type=quant_type))
            else:
                output = activ_quant.quantize(input, scale[0], room, quant_type=quant_type)
        else:
            if dual[0]:
                output = activ_quant.quantize(input.clone(), room, quant_type=quant_type)
                output.add_(activ_quant.quantize(input.add(-output), quant_type=quant_type))
            else:
                output = activ_quant.quantize(input, room, quant_type=quant_type)
            scale[0].data = output.scale.clone().data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        tracked_scale = ctx.tracked_scale
        dual = ctx.dual
        tracking = ctx.tracking
        quantize = ctx.quantize
        shortcut = ctx.shortcut
        distribution = ctx.distribution
        from .major import track_distribution
        if track_distribution:
            get_distribution(grad_output, distribution['backward'])
        if shortcut:
            from .major import shortcut_error_quant as error_quant
        else:
            from .major import error_quant
        if error_quant is None or quantize[1] is False:
            return grad_output, None, None, None, None, None, None, None, None, None
        room = None
        backward_scale = scale[1].clone()
        if tracking[1]:
            if dual[1]:
                grad_input = error_quant.quantize(grad_output.clone(), scale[1], room)
                grad_input.add_(error_quant.quantize(grad_output.add(-grad_input), backward_scale))
            else:
                grad_input = error_quant.quantize(grad_output, scale[1], room)
        else:
            if dual[1]:
                grad_input = error_quant.quantize(grad_output.clone(), room)
                grad_input.add_(error_quant.quantize(grad_output.add(-grad_input)))
            else:
                grad_input = error_quant.quantize(grad_output, room)
            scale[1].data = grad_input.scale.clone().data
        return grad_input, None, None, None, None, None, None, None, None, None

class QBlock(torch.nn.Module):
    def __init__(self, dual, fixed_scale, tracking, quantize, shortcut):
        super().__init__()
        self.dual = dual
        self.fixed_scale = fixed_scale
        self.tracking = tracking
        self.quantize = quantize
        self.shortcut = shortcut
        self.distribution = {'forward':{}, 'backward':{}}
        self.register_buffer('qtype', torch.tensor([-1], dtype=torch.int))
        for i in range(2):
            if fixed_scale[i] is None:
                self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))
            else:
                self.register_buffer('scale'+str(i+1), torch.tensor([fixed_scale[i]], dtype=torch.int))
            self.register_buffer('tracked_scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, input):
        scale = []
        tracked_scale = []
        for i in range(2):
            if self.fixed_scale[i] is None and self.training:
                scale.append(getattr(self, 'scale'+str(i+1)))
            else:
                scale.append(getattr(self, 'scale'+str(i+1)).clone())
            tracked_scale.append(getattr(self, 'tracked_scale'+str(i+1)))

        return qblock_func.apply(input, scale, tracked_scale, self.dual, self.training, self.tracking, self.quantize, self.shortcut, self.distribution, self.qtype)

class QLayer(torch.nn.Module):
    def __init__(self, module=None, function=None, dual=[False, False], fixed_scale=[None, None], tracking=[True, True], quantize=[True, True], shortcut=False):
        super().__init__()
        self.lp_module = module
        self.function = function
        self.qblock = QBlock(dual, fixed_scale, tracking, quantize, shortcut)
        
    def forward(self, a):
        if self.lp_module is not None:
            a = self.lp_module(a)
        if self.function is not None:
            a = self.function(a)
        return self.qblock(a)

class NQLayer(torch.nn.Module):
    def __init__(self, module=None, function=None):
        super().__init__()
        self.lp_module = module
        self.function = function
    
    def forward(self, a):
        if hasattr(a, 'scale'): scale = a.scale
        else: scale = None
        if self.lp_module is not None:
            a = self.lp_module(a)
        if self.function is not None:
            a = self.function(a)
        a.scale = scale
        return a

class wqblock_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, quantize):
        from .major import log4_trim_mantissa
        if quantize[0] is False:
            return input
        return log4_trim_mantissa(input, scale)
        # from .major import weight_quant
        # if weight_quant is None or quantize[0] is False:
        #     return input
        # return weight_quant.quantize(input.clone(), scale)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class WQBlock(torch.nn.Module):
    def __init__(self, quantize):
        super().__init__()
        self.quantize = quantize

    def forward(self, input, scale):
        return wqblock_func.apply(input, scale, self.quantize)

class WQLayer(torch.nn.Module):
    def __init__(self, module=None, function=None, dual=[False, False], fixed_scale=[None, None], tracking=[True, True], quantize=[True, True], shortcut=False):
        super().__init__()
        self.lp_module = module
        self.function = function
        self.qinput = WQBlock(quantize)
        self.qblock = QBlock(dual, fixed_scale, tracking, quantize, shortcut)

    def forward(self, input):
        if hasattr(input, 'scale'):
            scale = input.scale
        else:
            scale = None
            # print('Warning!! LQLayer input doesn\'t have scale')
        if self.lp_module is not None:
            a = self.qinput(input, scale)
            b = input.add(-a)
            a = self.lp_module(a)
            with torch.no_grad():
                bias_term = self.lp_module(torch.zeros_like(b))
                b = self.lp_module(b)
                b = b.add(-bias_term)
                # if self.lp_module.bias is not None:
                #     b.add_(-self.lp_module.bias)
            input = a.add(b)
        if self.function is not None:
            input = self.function(input)
        return self.qblock(input)

class qadd_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, scale, training, shortcut, distribution):
        from .major import track_distribution
        if track_distribution:
            get_distribution(a.add(b), distribution['forward'])
        if shortcut:
            from .major import shortcut_quant as activ_quant
        else:
            from .major import activ_quant
        if activ_quant is None:
            return a.add(b)
        if not training:
            scale[0] = scale[0].clone()
        output = activ_quant.quantize(a.add(b), scale[0])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None, None, None, None

class QAdd(torch.nn.Module):
    def __init__(self, shortcut=False):
        super().__init__()
        self.shortcut = shortcut
        self.register_buffer('scale1', torch.tensor([0], dtype=torch.int))
        self.distribution = {'forward':{}}

    def forward(self, a, b):
        scale = [self.scale1]
        return qadd_func.apply(a, b, scale, self.training, self.shortcut, self.distribution)
        
class qclone_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, scale, shortcut, distribution):
        b = a.clone().detach()
        if hasattr(a, 'scale') and a.scale is not None:
            b.scale = a.scale.clone().detach()
        ctx.scale = scale
        ctx.shortcut = shortcut
        ctx.distribution = distribution
        return a, b

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        scale = ctx.scale
        shortcut = ctx.shortcut
        distribution = ctx.distribution
        from .major import track_distribution
        if track_distribution:
            get_distribution(grad_output1, distribution['backward1'])
            get_distribution(grad_output2, distribution['backward2'])
        if shortcut:
            from .major import shortcut_error_quant as error_quant
        else:
            from .major import error_quant
        if error_quant is None:
            return grad_output1.add(grad_output2), None, None
        grad_output1 = error_quant.quantize(grad_output1, scale[0])
        grad_output2 = error_quant.quantize(grad_output2, scale[1])
        
        grad_input = grad_output1.add(grad_output2)
        return grad_input, None, None, None

class QClone(torch.nn.Module):
    def __init__(self, shortcut=False):
        super().__init__()
        self.shortcut = shortcut
        for i in range(2):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))
        self.distribution = {'backward1':{}, 'backward2':{}}

    def forward(self, a):
        scale = []
        for i in range(2):
            scale.append(getattr(self, 'scale'+str(i+1)))
        return qclone_func.apply(a, scale, self.shortcut, self.distribution)
        
class batch_norm2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, track_running_stats, momentum, eps):
        N = input.size(1) # channel
        input_scale = input.scale
        input = input.permute(0,2,3,1).contiguous()
        input_shape = input.shape
        input = input.view(-1,N)
        ctx.training = training
        if training:
            mu = input.mean(0)
            var = torch.var(input,0, unbiased=False)
            if track_running_stats:
                running_mean.data = running_mean.mul(1-momentum).add(mu.mul(momentum)).data
                running_var.data = running_var.mul(1-momentum).add(var.mul(momentum)).data
            sqrt = torch.sqrt(var+eps).reciprocal()
            mu = mu.mul(sqrt)
            weight_div_sqrt = weight.mul(sqrt).bfloat16().float()
            ctx.save_for_backward(mu, weight_div_sqrt, sqrt, input, input_scale)
            y = input.mul(weight_div_sqrt).add(bias.add(-mu*weight).bfloat16().float())
        else:
            y = input.mul(weight.div(torch.sqrt(running_var+eps)).bfloat16().float()).add(bias.add(-running_mean.div(torch.sqrt(running_var+eps)).mul(weight)).bfloat16().float())

        y = y.view(input_shape).permute(0,3,1,2).contiguous()
        return y
        
    @staticmethod
    def backward(ctx, grad_output):
        from .major import weight_quant
        # from .major import log4_trim_mantissa
        training = ctx.training
        if training:
            mu_div_sqrt, weight_div_sqrt, sqrt, input, input_scale = ctx.saved_tensors

            N = grad_output.size(1)
            grad_out = grad_output.permute(0,2,3,1).contiguous()
            grad_shape = grad_out.shape
            grad_out = grad_out.view(-1, N)

            grad_bias = torch.sum(grad_out, 0)
            # grad_weight = torch.sum(log4_trim_mantissa(input, input_scale)*grad_out, 0) * sqrt - grad_bias * mu_div_sqrt
            # grad_weight = torch.sum(weight_quant.quantize(input.clone(), input_scale)*grad_out, 0) * sqrt - grad_bias * mu_div_sqrt
            grad_weight = torch.sum(input*grad_out, 0) * sqrt - grad_bias * mu_div_sqrt
            grad_input = weight_div_sqrt * grad_out + (- weight_div_sqrt.mul(grad_weight*sqrt/grad_out.size(0)).bfloat16().float() * input \
                        + mu_div_sqrt.mul(grad_weight).add(-grad_bias).mul(weight_div_sqrt/grad_out.size(0)).bfloat16().float()).bfloat16().float()
            grad_input = grad_input.view(grad_shape).permute(0,3,1,2).contiguous()

            return grad_input, None, None, grad_weight, grad_bias, None, None, None, None
        else:
            return grad_output, None, None, None, None, None, None, None, None

class BatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.momentum = momentum
        
    def forward(self, input):
        self._check_input_dim(input)
        return batch_norm2d.apply(input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.track_running_stats, self.momentum, self.eps)

class QLayerNorm(torch.nn.Module):
    def __init__(self, d_hid, eps=1e-06, dual=[False, False], fixed_scale=[None, None], tracking=[True, True], quantize=[True, True], shortcut=False):
        super(QLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_hid))
        self.bias = torch.nn.Parameter(torch.zeros(d_hid))
        self.eps = eps
        self.qblock1 = QBlock(dual, fixed_scale, tracking, quantize, shortcut)
        self.qblock2 = QBlock(dual, fixed_scale, tracking, quantize, shortcut)
        
    def forward(self, input):
        mu = input.mean(dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, keepdim=True, unbiased=False)
        sqrt = torch.sqrt(var+self.eps).reciprocal()
        first_mul = sqrt
        first_add = -mu.mul(sqrt)
        with torch.no_grad():
            first_mul.data = first_mul.bfloat16().float().data
            first_add.data = first_add.bfloat16().float().data
        input = self.qblock1(input.mul(first_mul).add(first_add))
        second_mul = self.weight
        second_add = self.bias
        with torch.no_grad():
            second_mul.data = second_mul.bfloat16().float().data
            second_add.data = second_add.bfloat16().float().data
        input = self.qblock2(input.mul(second_mul).add(second_add))
        return input

class layer_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        mu = input.mean(dim=-1, keepdim=True).bfloat16()
        var = torch.var(input, dim=-1, keepdim=True, unbiased=False).bfloat16()
        sqrt = torch.sqrt(var+eps).reciprocal().bfloat16()
        input = input.bfloat16()
        normal = (input - mu) * sqrt
        weight = weight.bfloat16()
        output = normal * weight + bias.bfloat16()
        ctx.save_for_backward(normal, sqrt, weight)
        return output.float()
        
    @staticmethod
    def backward(ctx, grad_output):
        from .major import weight_quant
        # from .major import log4_trim_mantissa
        normal, sqrt, weight = ctx.saved_tensors
        weight_div_sqrt = sqrt.mul(weight)
        grad_output = grad_output.bfloat16()

        _, _, N = grad_output.shape
        grad_bias = torch.sum(grad_output.reshape(-1, N), 0)
        grad_weight = torch.sum(normal.mul(grad_output).reshape(-1, N), 0)
        grad_normal = grad_output * weight_div_sqrt
        grad_input = grad_normal - (torch.sum(grad_normal, 2, keepdim=True)+torch.sum(grad_normal*normal, 2, keepdim=True)*normal)/N
        
        return grad_input.float(), grad_weight.float(), grad_bias.float(), None

class LayerNormalization(torch.nn.Module):
    def __init__(self, d_hid, eps=1e-06):
        super(LayerNormalization, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_hid))
        self.beta = torch.nn.Parameter(torch.zeros(d_hid))
        self.eps = eps
        
    def forward(self, input):
        return layer_norm.apply(input, self.gamma, self.beta, self.eps)

class labelsmoothing_crossentropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, size_average, ignore_index, label_smoothing):
        from .major import activ_quant
        expo = input.exp().bfloat16()
        psum = expo.sum(dim=1, keepdim=True).pow(-1)
        expo = expo.mul(psum).float()
        log = -torch.log(expo)
        
        true_dist = torch.zeros_like(log)
        true_dist.fill_(label_smoothing / (input.size(1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1-label_smoothing)

        master_quant = quant.quant(quant.fp_format(exp_bit=6, man_bit=9))
        master = master_quant.quantize(true_dist.clone())
        
        ctx.save_for_backward(expo, target, master)
        ctx.ignore_index = ignore_index
        ctx.size_average = size_average
        
        loss = log.mul(true_dist).sum(dim=1)
        if ignore_index is not None:
            loss = torch.where(target != ignore_index, loss, torch.zeros(1).to(loss.device))
        if size_average:
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss
        
    @staticmethod
    def backward(ctx, grad_output):
        expo, target, true_dist = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        size_average = ctx.size_average
        grad_input = expo.add(-true_dist)
        if ignore_index is not None:
            grad_input.mul_(torch.where(target != ignore_index, 1, 0).unsqueeze(1))
        if size_average:
            grad_input.div_(expo.size(0))
        return grad_input, None, None, None, None, None        

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, size_average=True, ignore_index=None, label_smoothing=0):
        super().__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
            
    def forward(self, input, target):
        return labelsmoothing_crossentropy.apply(input, target, self.size_average, self.ignore_index, self.label_smoothing)

class soft_max(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, dim, scale):
        from .major import activ_quant
        expo = input.exp().bfloat16()
        if mask is not None:
            expo.masked_fill_(mask, 0)
        psum = expo.sum(dim=dim, keepdim=True).pow(-1)
        expo.mul_(psum)
        expo = expo.float()
        ctx.scale = scale
        ctx.dim = dim
        if activ_quant is None:
            ctx.save_for_backward(expo)
            return expo
        qp = activ_quant.quantize(expo, scale[0].mul(0))
        ctx.save_for_backward(qp)
        return qp
        
    @staticmethod
    def backward(ctx, grad_output):
        qp = ctx.saved_tensors[0]
        scale = ctx.scale
        dim = ctx.dim
        from .major import error_quant
        if error_quant is not None:
            grad_output = error_quant.quantize(grad_output, scale[1])
        output = grad_output.bfloat16().add(-torch.sum(qp.mul(grad_output), dim, keepdim=True).bfloat16()) * qp.bfloat16()
        output = output.float()
        return output, None, None, None
        

class Softmax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        for i in range(2):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))
            
    def forward(self, input, mask=None):
        if mask is not None:
            assert mask.size() == input.size()
        scale = []
        for i in range(2):
            scale.append(getattr(self, 'scale'+str(i+1)))
        return soft_max.apply(input, mask, self.dim, scale)
    
class lstm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, hx, bias, num_layers, dropout, training, bidirectional, batch_first, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        h = hx[0].clone()
        c = hx[1].clone()
        hidden_size = h.shape[2]
        if batch_first:
            input = input.permute(1,0,2)
        seq_len = input.shape[0]
        for l in range(num_layers):
            c_scale_max = scale[l*5].add(-2).clone()
            if l != 0:
                input = torch.stack(h_array)
                if dropout > 0 and training:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand/(1-dropout))
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    c_scale = scale[l*5].clone()
                    if bias:
                        flat = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2]).add(h[num_directions*l+j].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3]))
                    else:
                        flat = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1)).add(h[num_directions*l+j].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1)))
                    i = flat[:,0*hidden_size:1*hidden_size]
                    f = flat[:,1*hidden_size:2*hidden_size]
                    g = flat[:,2*hidden_size:3*hidden_size]
                    o = flat[:,3*hidden_size:4*hidden_size]

                    if activ_quant is None:
                        ii = torch.sigmoid(i)
                        ff = torch.sigmoid(f)
                        gg = torch.tanh(g)
                        oo = torch.sigmoid(o)
                        gate = torch.stack((ii,ff,gg,oo,h[num_directions*l+j],c[num_directions*l+j])).clone()
                        c[num_directions*l+j] = c[num_directions*l+j] * ff + ii * gg
                        tanhc = torch.tanh(c[num_directions*l+j])
                        h[num_directions*l+j] = oo * tanhc
                    else:
                        ii = activ_quant.quantize(torch.sigmoid(i), zero.clone())
                        ff = activ_quant.quantize(torch.sigmoid(f), zero.clone())
                        gg = activ_quant.quantize(torch.tanh(g), zero.clone())
                        oo = activ_quant.quantize(torch.sigmoid(o), zero.clone())
                        gate = torch.stack((ii,ff,gg,oo,h[num_directions*l+j],c[num_directions*l+j])).clone()
                        c[num_directions*l+j] = activ_quant.quantize(c[num_directions*l+j] * ff + ii * gg, c_scale)
                        tanhc = activ_quant.quantize(torch.tanh(c[num_directions*l+j]), zero.clone())
                        h[num_directions*l+j] = activ_quant.quantize(oo * tanhc, zero.clone())
                        c_scale_max = torch.max(c_scale_max, c_scale)
                    
                    gate_array.append(torch.cat((gate, torch.unsqueeze(tanhc, dim=0))).clone())

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j].clone())
            scale[l*5].data = c_scale_max.data
        output = torch.stack(h_array)
        if batch_first:
            output = output.permute(1,0,2)
        
        if training:
            gate_tensor = torch.stack(gate_array)
            if num_layers == 1:
                ctx.save_for_backward(gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.batch_first = batch_first
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h, c

    @staticmethod
    def backward(ctx, grad_output, grad_hidden, grad_cell):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        batch_first = ctx.batch_first
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[2]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        if batch_first:
            grad_output = grad_output.permute(1,0,2)
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            gc_scale_max = scale[l*5+1].add(-2).clone()
            ggate_scale_max = scale[l*5+2].add(-2).clone()
            gh_scale_max = scale[l*5+3].add(-2).clone()
            gx_scale_max = scale[l*5+4].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l]/(1-dropout))
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    gc_scale = scale[l*5+1].clone()
                    ggate_scale = scale[l*5+2].clone()
                    gh_scale = scale[l*5+3].clone()
                    gx_scale = scale[l*5+4].clone()

                    gate = gate_tensor[(l*num_directions+j)*seq_len+k]
                    ii, ff, gg, oo, h, c, cc = gate[0], gate[1], gate[2], gate[3], gate[4], gate[5], gate[6]
                    alpha = grad_hidden[num_directions*l+j].add(grad_output[(1-2*j)*k-j,:,hidden_size*j:hidden_size*(j+1)]) * oo * (1 - cc * cc) + grad_cell[num_directions*l+j]
                    gi = alpha * gg * ii * (1 - ii)
                    gf = alpha * c * ff * (1 - ff)
                    gg = alpha * ii * (1 - gg * gg)
                    go = grad_hidden[num_directions*l+j].add(grad_output[(1-2*j)*k-j,:,hidden_size*j:hidden_size*(j+1)]) * cc * oo * (1 - oo)
                    
                    if error_quant is None:
                        grad_cell[num_directions*l+j] = alpha * ff
                        g_gate = torch.cat((gi,gf,gg,go),dim=1)
                        grad_hidden[num_directions*l+j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1])
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                    else:
                        grad_cell[num_directions*l+j] = error_quant.quantize(alpha * ff, gc_scale)
                        g_gate = error_quant.quantize(torch.cat((gi,gf,gg,go),dim=1), ggate_scale)
                        grad_hidden[num_directions*l+j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]), gh_scale)
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[(1-2*j)*k-j] = error_quant.quantize(grad_input[(1-2*j)*k-j], gx_scale)
                        gc_scale_max = torch.max(gc_scale_max, gc_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)

                    grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[(1-2*j)*k-j]))
                    grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                    if bias:
                        grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
            scale[l*5+1].data = gc_scale_max.data
            scale[l*5+2].data = ggate_scale_max.data
            scale[l*5+3].data = gh_scale_max.data
            scale[l*5+4].data = gx_scale_max.data
        
        if batch_first:
            grad_input = grad_input.permute(1,0,2)
        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class packed_lstm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, batch_sizes, hx, bias, num_layers, dropout, training, bidirectional, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        gate_cat_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        h = hx[0].clone()
        c = hx[1].clone()
        hidden_size = h.shape[2]
        seq_len = batch_sizes.shape[0]
        for l in range(num_layers):
            c_scale_max = scale[l*5].add(-2).clone()
            if l != 0:
                input = torch.cat(h_array)
                if dropout > 0 and training:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand/(1-dropout))
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    c_scale = scale[l*5].clone()
                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if bias:
                        flat = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2]).add(h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3]))
                    else:
                        flat = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1)).add(h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1)))
                    i = flat[:,0*hidden_size:1*hidden_size]
                    f = flat[:,1*hidden_size:2*hidden_size]
                    g = flat[:,2*hidden_size:3*hidden_size]
                    o = flat[:,3*hidden_size:4*hidden_size]
                    
                    if activ_quant is None:
                        ii = torch.sigmoid(i)
                        ff = torch.sigmoid(f)
                        gg = torch.tanh(g)
                        oo = torch.sigmoid(o)
                        gate = torch.stack((ii,ff,gg,oo,h[num_directions*l+j][:current_batch_size],c[num_directions*l+j][:current_batch_size])).clone()
                        c[num_directions*l+j][:current_batch_size] = c[num_directions*l+j][:current_batch_size] * ff + ii * gg
                        tanhc = torch.tanh(c[num_directions*l+j][:current_batch_size])
                        h[num_directions*l+j][:current_batch_size] = oo * tanhc
                    else:
                        ii = activ_quant.quantize(torch.sigmoid(i), zero.clone())
                        ff = activ_quant.quantize(torch.sigmoid(f), zero.clone())
                        gg = activ_quant.quantize(torch.tanh(g), zero.clone())
                        oo = activ_quant.quantize(torch.sigmoid(o), zero.clone())
                        gate = torch.stack((ii,ff,gg,oo,h[num_directions*l+j][:current_batch_size],c[num_directions*l+j][:current_batch_size])).clone()
                        c[num_directions*l+j][:current_batch_size] = activ_quant.quantize(c[num_directions*l+j][:current_batch_size] * ff + ii * gg, c_scale)
                        tanhc = activ_quant.quantize(torch.tanh(c[num_directions*l+j][:current_batch_size]), zero.clone())
                        h[num_directions*l+j][:current_batch_size] = activ_quant.quantize(oo * tanhc, zero.clone())
                        c_scale_max = torch.max(c_scale_max, c_scale)

                    gate_array.append(torch.cat((gate, torch.unsqueeze(tanhc, dim=0))).clone())

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j][:current_batch_size].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j][:current_batch_size].clone())
                gate_cat_array.append(torch.cat(gate_array, dim=1))
                gate_array = []
            scale[l*5].data = c_scale_max.data
        output = torch.cat(h_array)
        
        if training:
            gate_tensor = torch.stack(gate_cat_array)
            if num_layers == 1:
                ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h, c

    @staticmethod
    def backward(ctx, grad_output, grad_hidden, grad_cell):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            batch_sizes, gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            batch_sizes, gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            batch_sizes, gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[1]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            gc_scale_max = scale[l*5+1].add(-2).clone()
            ggate_scale_max = scale[l*5+2].add(-2).clone()
            gh_scale_max = scale[l*5+3].add(-2).clone()
            gx_scale_max = scale[l*5+4].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l]/(1-dropout))
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    gc_scale = scale[l*5+1].clone()
                    ggate_scale = scale[l*5+2].clone()
                    gh_scale = scale[l*5+3].clone()
                    gx_scale = scale[l*5+4].clone()

                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if j is 1:
                        gate = gate_tensor[l*num_directions+j,:,batch_sizes.sum()-start_idx-current_batch_size:batch_sizes.sum()-start_idx,:]
                    else:
                        gate = gate_tensor[l*num_directions+j,:,start_idx:start_idx+current_batch_size,:]
                    ii, ff, gg, oo, h, c, cc = gate[0], gate[1], gate[2], gate[3], gate[4], gate[5], gate[6]
                    alpha = grad_hidden[num_directions*l+j][:current_batch_size].add(grad_output[start_idx:start_idx+current_batch_size,hidden_size*j:hidden_size*(j+1)]) * oo * (1 - cc * cc) + grad_cell[num_directions*l+j][:current_batch_size]
                    gi = alpha * gg * ii * (1 - ii)
                    gf = alpha * c * ff * (1 - ff)
                    gg = alpha * ii * (1 - gg * gg)
                    go = grad_hidden[num_directions*l+j][:current_batch_size].add(grad_output[start_idx:start_idx+current_batch_size,hidden_size*j:hidden_size*(j+1)]) * cc * oo * (1 - oo)
                    
                    if error_quant is None:
                        grad_cell[num_directions*l+j][:current_batch_size] = alpha * ff
                        g_gate = torch.cat((gi,gf,gg,go),dim=1)
                        grad_hidden[num_directions*l+j][:current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1])
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                    else:
                        grad_cell[num_directions*l+j][:current_batch_size] = error_quant.quantize(alpha * ff, gc_scale)
                        g_gate = error_quant.quantize(torch.cat((gi,gf,gg,go),dim=1), ggate_scale)
                        grad_hidden[num_directions*l+j][:current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]), gh_scale)
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_input[start_idx:start_idx+current_batch_size], gx_scale)
                        gc_scale_max = torch.max(gc_scale_max, gc_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)

                    grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[start_idx:start_idx+current_batch_size]))
                    grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                    if bias:
                        grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
            scale[l*5+1].data = gc_scale_max.data
            scale[l*5+2].data = ggate_scale_max.data
            scale[l*5+3].data = gh_scale_max.data
            scale[l*5+4].data = gx_scale_max.data
            
        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class LSTM(torch.nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, batch_first = False, dropout = 0., bidirectional = False):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        for i in range(num_layers*5):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, input, hx=None):  # noqa: F811
        if type(input) is tuple:
            input, _ = input
        scale = []
        for i in range(self.num_layers*5):
            scale.append(getattr(self, 'scale'+str(i+1)))

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = lstm_func.apply(input, scale, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first, *tuple(self._flat_weights))
        else:
            result = packed_lstm_func.apply(input, scale, batch_sizes, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, *tuple(self._flat_weights))
        
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)
'''
class gru_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, hx, bias, num_layers, dropout, training, bidirectional, batch_first, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        hidden_size = hx.shape[2]
        h = hx.clone()
        if batch_first:
            input = input.permute(1,0,2)
        seq_len = input.shape[0]
        for l in range(num_layers):
            h_scale_max = scale[l*5].add(-2).clone()
            flat_scale_max = scale[l*5+1].add(-2).clone()
            if l != 0:
                input = torch.stack(h_array)
                if dropout > 0:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand)
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    h_scale = scale[l*5].clone()
                    flat_scale1 = scale[l*5+1].clone()
                    flat_scale2 = scale[l*5+1].clone()
                    flat_scale3 = scale[l*5+1].clone()
                    flat_scale4 = scale[l*5+1].clone()
                    if bias:
                        flat_x = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2])
                        flat_h = h[num_directions*l+j].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3])
                    else:
                        flat_x = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1))
                        flat_h = h[num_directions*l+j].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1))
                    
                    if activ_quant is None:
                        r = flat_x[:,0*hidden_size:1*hidden_size].add(flat_h[:,0*hidden_size:1*hidden_size])
                        z = flat_x[:,1*hidden_size:2*hidden_size].add(flat_h[:,1*hidden_size:2*hidden_size])
                        rr = torch.sigmoid(r)
                        zz = torch.sigmoid(z)
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = torch.tanh(n)
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j].mul_(zz).add_((1 - zz) * nn)
                    else:
                        r = activ_quant.quantize(flat_x[:,0*hidden_size:1*hidden_size].add(flat_h[:,0*hidden_size:1*hidden_size]), flat_scale1)
                        z = activ_quant.quantize(flat_x[:,1*hidden_size:2*hidden_size].add(flat_h[:,1*hidden_size:2*hidden_size]), flat_scale2)
                        rr = activ_quant.quantize(torch.sigmoid(r), zero.clone())
                        zz = activ_quant.quantize(torch.sigmoid(z), zero.clone())
                        flat_h[:,2*hidden_size:3*hidden_size] = activ_quant.quantize(flat_h[:,2*hidden_size:3*hidden_size], flat_scale3)
                        n = activ_quant.quantize(flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size]), flat_scale4)
                        nn = activ_quant.quantize(torch.tanh(n), zero.clone())
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j] = activ_quant.quantize(h[num_directions*l+j].mul(zz).add((1 - zz) * nn), h_scale)
                        h_scale_max = torch.max(h_scale_max, h_scale)
                        flat_scale_max = torch.max(torch.max(torch.max(torch.max(flat_scale_max,flat_scale1),flat_scale2),flat_scale3),flat_scale4)

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j].clone())
            scale[l*5].data = h_scale_max.data
            scale[l*5+1].data = flat_scale_max.data
        output = torch.stack(h_array)
        if batch_first:
            output = output.permute(1,0,2)
        
        if training:
            gate_tensor = torch.stack(gate_array)
            if num_layers == 1:
                ctx.save_for_backward(gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.batch_first = batch_first
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h
    
    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        batch_first = ctx.batch_first
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[2]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        if batch_first:
            grad_output = grad_output.permute(1,0,2)
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            ggate_scale_max = scale[l*5+2].add(-2).clone()
            gx_scale_max = scale[l*5+3].add(-2).clone()
            gh_scale_max = scale[l*5+4].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l])
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    ggate_scale = scale[l*5+2].clone()
                    gx_scale = scale[l*5+3].clone()
                    gh_scale = scale[l*5+4].clone()
                    gate = gate_tensor[(l*num_directions+j)*seq_len+k]
                    rr, zz, nn, h, flat_h = gate[0], gate[1], gate[2], gate[3], gate[4]
                    alpha = grad_hidden[num_directions*l+j].add(grad_output[(1-2*j)*k-j,:,hidden_size*j:hidden_size*(j+1)])
                    gn = alpha * (1 - zz) * (1 - nn * nn)
                    gz = alpha * (h - nn) * zz * (1 - zz)
                    gr = gn * flat_h * rr * (1 - rr)

                    if error_quant is None:
                        g_gate = torch.cat((gr,gz,gn),dim=1)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[(1-2*j)*k-j]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:   
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        g_gate[:,2*hidden_size:].mul_(rr)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz
                    else:
                        g_gate = error_quant.quantize(torch.cat((gr,gz,gn),dim=1), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[(1-2*j)*k-j]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:   
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[(1-2*j)*k-j] = error_quant.quantize(grad_input[(1-2*j)*k-j], gx_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        ggate_scale = scale[l*5+2].clone()
                        g_gate[:,2*hidden_size:] = error_quant.quantize(g_gate[:,2*hidden_size:].mul(rr), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz, gh_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
            scale[l*5+2].data = ggate_scale_max.data
            scale[l*5+3].data = gx_scale_max.data
            scale[l*5+4].data = gh_scale_max.data
        
        if batch_first:
            grad_input = grad_input.permute(1,0,2)
        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class packed_gru_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, batch_sizes, hx, bias, num_layers, dropout, training, bidirectional, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        gate_cat_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        hidden_size = hx.shape[2]
        h = hx.clone()
        seq_len = batch_sizes.shape[0]
        for l in range(num_layers):
            h_scale_max = scale[l*4].add(-2).clone()
            if l != 0:
                input = torch.cat(h_array)
                if dropout > 0:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand)
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    h_scale = scale[l*4].clone()
                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if bias:
                        flat_x = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2])
                        flat_h = h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3])
                    else:
                        flat_x = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1))
                        flat_h = h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1))
                    r = flat_x[:,0*hidden_size:1*hidden_size].add(flat_h[:,0*hidden_size:1*hidden_size])
                    z = flat_x[:,1*hidden_size:2*hidden_size].add(flat_h[:,1*hidden_size:2*hidden_size])
                    
                    if activ_quant is None:
                        rr = torch.sigmoid(r)
                        zz = torch.sigmoid(z)
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = torch.tanh(n)
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j][:current_batch_size],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j][:current_batch_size].mul_(zz).add_((1 - zz) * nn)
                    else:
                        rr = activ_quant.quantize(torch.sigmoid(r), zero.clone())
                        zz = activ_quant.quantize(torch.sigmoid(z), zero.clone())
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = activ_quant.quantize(torch.tanh(n), zero.clone())
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j][:current_batch_size],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j][:current_batch_size] = activ_quant.quantize(h[num_directions*l+j][:current_batch_size].mul(zz).add((1 - zz) * nn), h_scale)
                        h_scale_max = torch.max(h_scale_max, h_scale)

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j][:current_batch_size].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j][:current_batch_size].clone())
                gate_cat_array.append(torch.cat(gate_array, dim=1))
                gate_array = []
            scale[l*4].data = h_scale_max.data
        output = torch.cat(h_array)
        
        if training:
            gate_tensor = torch.stack(gate_cat_array)
            if num_layers == 1:
                ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h

    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            batch_sizes, gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            batch_sizes, gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            batch_sizes, gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[1]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            ggate_scale_max = scale[l*4+1].add(-2).clone()
            gx_scale_max = scale[l*4+2].add(-2).clone()
            gh_scale_max = scale[l*4+3].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l])
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    ggate_scale = scale[l*4+1].clone()
                    gx_scale = scale[l*4+2].clone()
                    gh_scale = scale[l*4+3].clone()
                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if j is 1:
                        gate = gate_tensor[l*num_directions+j,:,batch_sizes.sum()-start_idx-current_batch_size:batch_sizes.sum()-start_idx,:]
                    else:
                        gate = gate_tensor[l*num_directions+j,:,start_idx:start_idx+current_batch_size,:]
                    rr, zz, nn, h, flat_h = gate[0], gate[1], gate[2], gate[3], gate[4]
                    alpha = grad_hidden[num_directions*l+j][:current_batch_size].add(grad_output[start_idx:start_idx+current_batch_size,hidden_size*j:hidden_size*(j+1)])
                    gn = alpha * (1 - zz) * (1 - nn * nn)
                    gz = alpha * (h - nn) * zz * (1 - zz)
                    gr = gn * flat_h * rr * (1 - rr)

                    if error_quant is None:
                        g_gate = torch.cat((gr,gz,gn),dim=1)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[start_idx:start_idx+current_batch_size]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        g_gate[:,2*hidden_size:].mul_(rr)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j][:current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz
                    else:
                        g_gate = error_quant.quantize(torch.cat((gr,gz,gn),dim=1), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[start_idx:start_idx+current_batch_size]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:   
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_input[start_idx:start_idx+current_batch_size], gx_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        ggate_scale = scale[l*4+1].clone()
                        g_gate[:,2*hidden_size:] = error_quant.quantize(g_gate[:,2*hidden_size:].mul(rr), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j][:current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz, gh_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
            scale[l*4+1].data = ggate_scale_max.data
            scale[l*4+2].data = gx_scale_max.data
            scale[l*4+3].data = gh_scale_max.data

        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class GRU(torch.nn.GRU):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, batch_first = False, dropout = 0., bidirectional = False):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        for i in range(num_layers*5):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, input, hx=None):  # noqa: F811
        if type(input) is tuple:
            input, _ = input
        scale = []
        for i in range(self.num_layers*5):
            scale.append(getattr(self, 'scale'+str(i+1)))

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = gru_func.apply(input, scale, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first, *tuple(self._flat_weights))
        else:
            result = packed_gru_func.apply(input, scale, batch_sizes, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, *tuple(self._flat_weights))

        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)
'''
# origin version
class gru_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, hx, bias, num_layers, dropout, training, bidirectional, batch_first, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        hidden_size = hx.shape[2]
        h = hx.clone()
        if batch_first:
            input = input.permute(1,0,2)
        seq_len = input.shape[0]
        for l in range(num_layers):
            h_scale_max = scale[l*4].add(-2).clone()
            if l != 0:
                input = torch.stack(h_array)
                if dropout > 0 and training:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand/(1-dropout))
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    h_scale = scale[l*4].clone()
                    if bias:
                        flat_x = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2])
                        flat_h = h[num_directions*l+j].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3])
                    else:
                        flat_x = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1))
                        flat_h = h[num_directions*l+j].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1))
                    r = flat_x[:,0*hidden_size:1*hidden_size].add(flat_h[:,0*hidden_size:1*hidden_size])
                    z = flat_x[:,1*hidden_size:2*hidden_size].add(flat_h[:,1*hidden_size:2*hidden_size])
                    
                    if activ_quant is None:
                        rr = torch.sigmoid(r)
                        zz = torch.sigmoid(z)
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = torch.tanh(n)
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j].mul_(zz).add_((1 - zz) * nn)
                    else:
                        rr = activ_quant.quantize(torch.sigmoid(r), zero.clone())
                        zz = activ_quant.quantize(torch.sigmoid(z), zero.clone())
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = activ_quant.quantize(torch.tanh(n), zero.clone())
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j] = activ_quant.quantize(h[num_directions*l+j].mul(zz).add((1 - zz) * nn), h_scale)
                        h_scale_max = torch.max(h_scale_max, h_scale)

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j].clone())
            scale[l*4].data = h_scale_max.data
        output = torch.stack(h_array)
        if batch_first:
            output = output.permute(1,0,2)
        
        if training:
            gate_tensor = torch.stack(gate_array)
            if num_layers == 1:
                ctx.save_for_backward(gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.batch_first = batch_first
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h
    
    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        batch_first = ctx.batch_first
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[2]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        if batch_first:
            grad_output = grad_output.permute(1,0,2)
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            ggate_scale_max = scale[l*4+1].add(-2).clone()
            gx_scale_max = scale[l*4+2].add(-2).clone()
            gh_scale_max = scale[l*4+3].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l]/(1-dropout))
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    ggate_scale = scale[l*4+1].clone()
                    gx_scale = scale[l*4+2].clone()
                    gh_scale = scale[l*4+3].clone()
                    gate = gate_tensor[(l*num_directions+j)*seq_len+k]
                    rr, zz, nn, h, flat_h = gate[0], gate[1], gate[2], gate[3], gate[4]
                    alpha = grad_hidden[num_directions*l+j].add(grad_output[(1-2*j)*k-j,:,hidden_size*j:hidden_size*(j+1)])
                    gn = alpha * (1 - zz) * (1 - nn * nn)
                    gz = alpha * (h - nn) * zz * (1 - zz)
                    gr = gn * flat_h * rr * (1 - rr)

                    if error_quant is None:
                        g_gate = torch.cat((gr,gz,gn),dim=1)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[(1-2*j)*k-j]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:   
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        g_gate[:,2*hidden_size:].mul_(rr)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz
                    else:
                        g_gate = error_quant.quantize(torch.cat((gr,gz,gn),dim=1), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[(1-2*j)*k-j]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:   
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[(1-2*j)*k-j] = error_quant.quantize(grad_input[(1-2*j)*k-j], gx_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        ggate_scale = scale[l*4+1].clone()
                        g_gate[:,2*hidden_size:] = error_quant.quantize(g_gate[:,2*hidden_size:].mul(rr), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz, gh_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
            scale[l*4+1].data = ggate_scale_max.data
            scale[l*4+2].data = gx_scale_max.data
            scale[l*4+3].data = gh_scale_max.data
        
        if batch_first:
            grad_input = grad_input.permute(1,0,2)
        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class packed_gru_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, batch_sizes, hx, bias, num_layers, dropout, training, bidirectional, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        gate_cat_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        hidden_size = hx.shape[2]
        h = hx.clone()
        seq_len = batch_sizes.shape[0]
        for l in range(num_layers):
            h_scale_max = scale[l*4].add(-2).clone()
            if l != 0:
                input = torch.cat(h_array)
                if dropout > 0 and training:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand/(1-dropout))
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    h_scale = scale[l*4].clone()
                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if bias:
                        flat_x = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2])
                        flat_h = h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3])
                    else:
                        flat_x = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1))
                        flat_h = h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1))
                    r = flat_x[:,0*hidden_size:1*hidden_size].add(flat_h[:,0*hidden_size:1*hidden_size])
                    z = flat_x[:,1*hidden_size:2*hidden_size].add(flat_h[:,1*hidden_size:2*hidden_size])
                    
                    if activ_quant is None:
                        rr = torch.sigmoid(r)
                        zz = torch.sigmoid(z)
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = torch.tanh(n)
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j][:current_batch_size],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j][:current_batch_size].mul_(zz).add_((1 - zz) * nn)
                    else:
                        rr = activ_quant.quantize(torch.sigmoid(r), zero.clone())
                        zz = activ_quant.quantize(torch.sigmoid(z), zero.clone())
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = activ_quant.quantize(torch.tanh(n), zero.clone())
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j][:current_batch_size],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j][:current_batch_size] = activ_quant.quantize(h[num_directions*l+j][:current_batch_size].mul(zz).add((1 - zz) * nn), h_scale)
                        h_scale_max = torch.max(h_scale_max, h_scale)

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j][:current_batch_size].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j][:current_batch_size].clone())
                gate_cat_array.append(torch.cat(gate_array, dim=1))
                gate_array = []
            scale[l*4].data = h_scale_max.data
        output = torch.cat(h_array)
        
        if training:
            gate_tensor = torch.stack(gate_cat_array)
            if num_layers == 1:
                ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h

    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            batch_sizes, gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            batch_sizes, gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            batch_sizes, gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[1]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            ggate_scale_max = scale[l*4+1].add(-2).clone()
            gx_scale_max = scale[l*4+2].add(-2).clone()
            gh_scale_max = scale[l*4+3].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l]/(1-dropout))
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    ggate_scale = scale[l*4+1].clone()
                    gx_scale = scale[l*4+2].clone()
                    gh_scale = scale[l*4+3].clone()
                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if j is 1:
                        gate = gate_tensor[l*num_directions+j,:,batch_sizes.sum()-start_idx-current_batch_size:batch_sizes.sum()-start_idx,:]
                    else:
                        gate = gate_tensor[l*num_directions+j,:,start_idx:start_idx+current_batch_size,:]
                    rr, zz, nn, h, flat_h = gate[0], gate[1], gate[2], gate[3], gate[4]
                    alpha = grad_hidden[num_directions*l+j][:current_batch_size].add(grad_output[start_idx:start_idx+current_batch_size,hidden_size*j:hidden_size*(j+1)])
                    gn = alpha * (1 - zz) * (1 - nn * nn)
                    gz = alpha * (h - nn) * zz * (1 - zz)
                    gr = gn * flat_h * rr * (1 - rr)

                    if error_quant is None:
                        g_gate = torch.cat((gr,gz,gn),dim=1)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[start_idx:start_idx+current_batch_size]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        g_gate[:,2*hidden_size:].mul_(rr)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j][:current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz
                    else:
                        g_gate = error_quant.quantize(torch.cat((gr,gz,gn),dim=1), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[start_idx:start_idx+current_batch_size]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:   
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_input[start_idx:start_idx+current_batch_size], gx_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        ggate_scale = scale[l*4+1].clone()
                        g_gate[:,2*hidden_size:] = error_quant.quantize(g_gate[:,2*hidden_size:].mul(rr), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j][:current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz, gh_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
            scale[l*4+1].data = ggate_scale_max.data
            scale[l*4+2].data = gx_scale_max.data
            scale[l*4+3].data = gh_scale_max.data

        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class GRU(torch.nn.GRU):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, batch_first = False, dropout = 0., bidirectional = False):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        for i in range(num_layers*4):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, input, hx=None):  # noqa: F811
        if type(input) is tuple:
            input, _ = input
        scale = []
        for i in range(self.num_layers*4):
            scale.append(getattr(self, 'scale'+str(i+1)))

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = gru_func.apply(input, scale, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first, *tuple(self._flat_weights))
        else:
            result = packed_gru_func.apply(input, scale, batch_sizes, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, *tuple(self._flat_weights))

        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)

# not recommended to use
# class ConvBn2d(torch.nn.Module):
#     def __init__(self, conv, bn, dual=[False, False], fixed_scale=[None, None], last=False, tracking=[True, True]):
#         super().__init__()
#         self.cb_conv = conv
#         self.cb_bn = bn
#         # self.cb_bn.momentum = 0.01
#         self.last = last
#         self.merge = False
#         self.qtrain = False
#         self.qblock = QBlock(dual, fixed_scale, tracking, bn=True)
#         self.qfused_weight = torch.empty_like(self.cb_conv.weight)
#         self.qfused_bias = torch.empty_like(self.cb_bn.bias)
#         self.qfused_initialized = False

#     def forward(self, inputs):
#         if type(inputs) is not tuple:
#             aa, b = inputs, torch.zeros(1).to(inputs.device)
#         else:
#             aa, b = inputs
#         if self.merge:
#             a = self.cb_conv(aa)
#         else:
#             temp = self.cb_conv(aa)
#             a = self.cb_bn(temp)
#             if self.qtrain:
#                 from .major import weight_quant, hysteresis_update
#                 var_b = torch.var(temp, dim=[0,2,3], unbiased=False)
#                 std_b = torch.sqrt(var_b+self.cb_bn.eps)
#                 mu_b = torch.mean(temp, dim=[0,2,3])
#                 var = self.cb_bn.running_var
#                 std = torch.sqrt(var+self.cb_bn.eps)
#                 mu = self.cb_bn.running_mean
#                 with torch.no_grad():
#                     mul_val = self.cb_bn.weight.div(std).reshape(-1,1,1,1)
#                     fused_weight = self.cb_conv.weight.mul(mul_val)
#                     fused_bias = self.cb_bn.bias.add(-self.cb_bn.weight.mul(mu_b).div(std_b))
#                     if hysteresis_update is False or self.qfused_initialized is False:
#                         self.qfused_weight.data = weight_quant.quantize(fused_weight.clone()).data
#                         self.qfused_bias.data = weight_quant.quantize(fused_bias.clone()).data
#                         self.qfused_initialized = True
#                     else:
#                         self.qfused_weight.data = weight_quant.hysteresis(self.qfused_weight, fused_weight.clone()).data
#                         self.qfused_bias.data = weight_quant.hysteresis(self.qfused_bias, fused_bias.clone()).data
#                     qweight_err = self.qfused_weight.add(-fused_weight).div(mul_val)
#                     qbias_err = self.qfused_bias.add(-fused_bias).reshape(1,-1,1,1)
#                     temp_err = torch.nn.functional.conv2d(aa, qweight_err, bias=None, stride=self.cb_conv.stride, padding=self.cb_conv.padding, dilation=self.cb_conv.dilation, groups=self.cb_conv.groups)
#                     a_err = temp_err.div(std_b.reshape(1,-1,1,1)).add(qbias_err)
#                 a = a + a_err

#         outputs = self.qblock((a,b))
#         if self.last:
#             return outputs[0]
#         else:
#             return outputs

#     def merge_bn(self, from_qfused_buffer=False):
#         if self.merge:
#             return
#         from .major import weight_quant
#         div_val = torch.sqrt(self.cb_bn.running_var.add(self.cb_bn.eps))
#         mul_val = self.cb_bn.weight.div(div_val)
#         add_val = self.cb_bn.bias.add(-self.cb_bn.running_mean.mul(mul_val))
#         if weight_quant is None or self.qtrain == False:
#             if from_qfused_buffer:
#                 self.cb_conv.weight.data = self.qfused_weight.data
#             else:
#                 self.cb_conv.weight.data = self.cb_conv.weight.mul(mul_val.reshape(-1,1,1,1)).data
#             if self.cb_conv.bias is None:
#                 self.cb_conv.bias = torch.nn.Parameter(add_val)
#             else:
#                 self.cb_conv.bias.data = self.cb_conv.bias.mul(mul_val).add(add_val).data
#         else:
#             if from_qfused_buffer:
#                 self.cb_conv.weight.data = self.qfused_weight.data
#             else:
#                 self.cb_conv.weight.data = weight_quant.quantize(self.cb_conv.weight.mul(mul_val.reshape(-1,1,1,1))).data
#             if self.cb_conv.bias is None:
#                 self.cb_conv.bias = torch.nn.Parameter(weight_quant.quantize(add_val))
#             else:
#                 self.cb_conv.bias.data = weight_quant.quantize(self.cb_conv.bias.mul(mul_val).add(add_val)).data
#         self.merge = True

#     def set_qtrain(self, qtrain=True):
#         self.qtrain = qtrain
    
#     def set_merge(self, merge=True):
#         self.merge = merge
