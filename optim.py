import torch
import torch.nn as nn
import onnx
import math
import pdb
class SGD(torch.optim.SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                weight_decay=0, nesterov=False, weight_quantize=True, quant=None, bias=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.weight_quantize = weight_quantize
        self.quant = quant
        self.bias = bias

    @torch.no_grad()
    def step(self, closure=None, lr=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.bias:
            from .major import bias_quant as weight_quant
        else:
            from .major import weight_quant
        from .major import grad_quant, master_quant, hysteresis_update
        if self.quant is not None:
            weight_quant = self.quant

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            param = torch.tensor((group['weight_decay'],group['momentum'], group['lr']), device=group['params'][0].device)
            # if master_quant is not None:
            #     param = master_quant.quantize(param)
            if grad_quant is not None:
                param = grad_quant.quantize(param)
            weight_decay = param[0].item()
            momentum = param[1].item()
            if lr is None:
                lr = param[2].item()

            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                d_p = p.grad.data

                # weight gradient quantize
                if 'd_p_scale' not in param_state:
                    d_p_scale = param_state['d_p_scale'] = d_p.abs().max().log2().floor().int()
                else:
                    d_p_scale = param_state['d_p_scale']
                if grad_quant is not None:
                    d_p.data = grad_quant.quantize(d_p, d_p_scale).data

                # master quantize
                if 'master_value' not in param_state:
                    master = param_state['master_value'] = torch.clone(p.data).detach()
                    if self.weight_quantize and weight_quant is not None:
                        p.data = weight_quant.quantize(p).data
                else:
                    master = param_state['master_value']
                if 'master_scale' not in param_state:
                    master_scale = param_state['master_scale'] = [torch.tensor([0], dtype=torch.int).to(master.device),
                                                                torch.tensor([0], dtype=torch.int).to(master.device),
                                                                torch.tensor([0], dtype=torch.int).to(master.device)]
                else:
                    master_scale = param_state['master_scale']
                
                if master_quant is not None:
                    d_p.data = master_quant.quantize(d_p.add(p.data, alpha=weight_decay), master_scale[0]).data
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.data = master_quant.quantize(d_p.add(buf, alpha=momentum), master_scale[1]).data
                    master.data = master_quant.quantize(master.add(buf, alpha=-lr), master_scale[2]).data
                else:
                    d_p.data = d_p.add(p.data, alpha=weight_decay).data
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.data = d_p.add(buf, alpha=momentum).data
                    master.data = master.add(buf, alpha=-lr).data

                if self.weight_quantize and (weight_quant is not None):
                    # weight quantize
                    if 'p_scale' not in param_state:
                        if p.dim() is 4 and weight_quant.ch_wise:
                            scale = p.abs().reshape(p.shape[0],-1).max(1)[0]
                        else:
                            scale = p.abs().max()
                        scale = torch.where(scale>0, scale.log2().floor(), scale)
                        p_scale = param_state['p_scale'] = scale.int()
                    else:
                        p_scale = param_state['p_scale'] = param_state['p_scale'].int()
                    
                    if hysteresis_update:
                        p.data = weight_quant.hysteresis(p, master.clone(), p_scale).data
                    else:
                        p.data = weight_quant.quantize(master.clone(), p_scale).data
                else:
                    p.data = master.data
        return loss

    def scale_to_int(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'd_p_scale' in param_state:
                    param_state['d_p_scale'] = param_state['d_p_scale'].int()
                if 'p_scale' in param_state:
                    param_state['p_scale'] = param_state['p_scale'].int()
    
    def clear_optim(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                del(param_state['master_value'])
                del(param_state['momentum_buffer'])

class RMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, weight_quantize=True, quant=None, bias=False):
        super().__init__(params, lr, alpha, eps, weight_decay, momentum, centered)
        self.weight_quantize = weight_quantize
        self.quant = quant
        self.bias = bias

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.bias:
            from .major import bias_quant as weight_quant
        else:
            from .major import weight_quant
        from .major import grad_quant, master_quant, hysteresis_update
        if self.quant is not None:
            weight_quant = self.quant

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            param = torch.tensor((group['weight_decay'],group['momentum'], group['lr'], group['eps']), device=group['params'][0].device)
            if master_quant is not None:
                param = master_quant.quantize(param)
            weight_decay = param[0].item()
            momentum = param[1].item()
            lr = param[2].item()
            eps = param[3].item()

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # weight gradient quantize
                if 'grad_scale' not in state:
                    grad_scale = state['grad_scale'] = grad.abs().max().log2().floor().int()
                else:
                    grad_scale = state['grad_scale']
                if grad_quant is not None:
                    grad.data = grad_quant.quantize(grad, grad_scale).data

                # master quantize
                if 'master_value' not in state:
                    master = state['master_value'] = torch.clone(p.data).detach()
                    if self.weight_quantize and weight_quant is not None:
                        p.data = weight_quant.quantize(p).data
                else:
                    master = state['master_value']
                if 'master_scale' not in state:
                    master_scale = state['master_scale'] = [torch.tensor([0], dtype=torch.int).to(master.device),
                                                            torch.tensor([0], dtype=torch.int).to(master.device),
                                                            torch.tensor([0], dtype=torch.int).to(master.device),
                                                            torch.tensor([0], dtype=torch.int).to(master.device),
                                                            torch.tensor([0], dtype=torch.int).to(master.device)]
                else:
                    master_scale = state['master_scale']

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1
                
                if master_quant is not None:
                    if weight_decay != 0:
                        grad = master_quant.quantize(grad.add(p, alpha=weight_decay), master_scale[0])

                    square_avg.data = master_quant.quantize(square_avg.mul(alpha).addcmul(grad, grad, value=1 - alpha), master_scale[1]).data

                    if group['centered']:
                        grad_avg = state['grad_avg']
                        grad_avg.data = master_quant.quantize(grad_avg.mul(alpha).add(grad, alpha=1 - alpha), master_scale[2]).data
                        avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
                    else:
                        avg = square_avg.sqrt().add_(eps)
                    avg.data = master_quant.quantize(avg, master_scale[3]).data

                    if momentum > 0:
                        buf = state['momentum_buffer']
                        buf.data = master_quant.quantize(buf.mul(momentum).addcdiv(grad, avg), master_scale[4]).data
                        master.add_(buf, alpha=-lr)
                    else:
                        master.addcdiv_(grad, avg, value=-lr)
                    master.data = master_quant.quantize(master, master_scale[5]).data
                else:
                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)

                    square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                    if group['centered']:
                        grad_avg = state['grad_avg']
                        grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                        avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
                    else:
                        avg = square_avg.sqrt().add_(eps)

                    if momentum > 0:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).addcdiv_(grad, avg)
                        master.add_(buf, alpha=-lr)
                    else:
                        master.addcdiv_(grad, avg, value=-lr)

                # weight quantize
                if 'p_scale' not in state:
                    scale = p.abs().max()
                    if scale == 0:
                        p_scale = state['p_scale'] = scale.int()
                    else:
                        p_scale = state['p_scale'] = scale.log2().floor().int()
                else:
                    p_scale = state['p_scale']
                if self.weight_quantize and (weight_quant is not None):
                    if hysteresis_update:
                        p.data = weight_quant.hysteresis(p, master.clone(), p_scale).data
                    else:
                        p.data = weight_quant.quantize(master.clone(), p_scale).data
                else:
                    p.data = master.data
        return loss

class fair_Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        quantize,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        bias=False,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(fair_Adam, self).__init__(params, defaults)
        self.weight_quantize = quantize
        self.bias = bias
    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.bias:
            from .major import bias_quant as weight_quant
        else:
            from .major import weight_quant
        from .major import grad_quant, master_quant, hysteresis_update

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group.get("amsgrad", False)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    state['master_value'] = torch.clone(p_data_fp32.data).detach()
                    if self.weight_quantize and weight_quant is not None:
                        p_data_fp32.data = weight_quant.quantize(p_data_fp32).data
                    state['d_p_scale'] = grad.abs().max().log2().floor().int()
                    state['master_scale'] = p_data_fp32.abs().max().log2().floor().int()
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(
                            p_data_fp32
                        )
                # grad quantize
                d_p_scale = state['d_p_scale']
                if grad_quant is not None:
                    grad.data = grad_quant.quantize(grad, d_p_scale).data
                # master quantize
                master = state['master_value']
                master_scale = state['master_scale']
                if master_quant is not None:
                    master.data = master_quant.quantize(master, master_scale).data

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    master.add_(
                        master, alpha=-group["weight_decay"] * group["lr"]
                    )

                master.addcdiv_(exp_avg, denom, value=-step_size)
                # weight quantize
                if self.weight_quantize and (weight_quant is not None):
                    if hysteresis_update:
                        p_data_fp32.copy_(weight_quant.hysteresis(p_data_fp32, master.clone()))
                    else:
                        p_data_fp32.copy_(weight_quant.quantize(master.clone()))
                else:
                    p_data_fp32.copy_(master)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


class Adam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, weight_quantize=True, quant=None, bias=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.weight_quantize = weight_quantize
        self.quant = quant
        self.bias = bias

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.bias:
            from .major import bias_quant as weight_quant
        else:
            from .major import weight_quant
        from .major import hysteresis_update
        if self.quant is not None:
            weight_quant = self.quant

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            amsgrad = group['amsgrad']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['master_value'] = torch.clone(p.data).detach()
                        if self.weight_quantize and weight_quant is not None:
                            p.data = weight_quant.quantize(p).data

                    master = state['master_value']

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    step = state['step']

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    if weight_decay != 0:
                        grad.data = grad.add(master, alpha=weight_decay).data

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    step_size = lr / bias_correction1

                    master.addcdiv_(exp_avg, denom, value=-step_size)

                    # weight quantize
                    if self.weight_quantize and (weight_quant is not None):
                        # weight quantize
                        if 'p_scale' not in state:
                            scale = p.abs().max()
                            scale = torch.where(scale>0, scale.log2().floor(), scale)
                            p_scale = state['p_scale'] = scale.int()
                        else:
                            p_scale = state['p_scale'] = state['p_scale'].int()
                            
                        if hysteresis_update:
                            p.data = weight_quant.hysteresis(p, master.clone(), p_scale).data
                        else:
                            p.data = weight_quant.quantize(master.clone(), p_scale).data
                    else:
                        p.data = master.data
                
        return loss

class AdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, weight_quantize=True, quant=None, bias=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.weight_quantize = weight_quantize
        self.quant = quant
        self.bias = bias

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.bias:
            from .major import bias_quant as weight_quant
        else:
            from .major import weight_quant
        from .major import hysteresis_update
        if self.quant is not None:
            weight_quant = self.quant
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['master_value'] = torch.clone(p.data).detach()
                    if self.weight_quantize and weight_quant is not None:
                        p.data = weight_quant.quantize(p).data

                master = state['master_value']

                # Perform stepweight decay
                master.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                master.addcdiv_(exp_avg, denom, value=-step_size)

                # weight quantize
                if self.weight_quantize and (weight_quant is not None):
                    if hysteresis_update:
                        p.data = weight_quant.hysteresis(p, master.clone()).data
                    else:
                        p.data = weight_quant.quantize(master.clone()).data
                else:
                    p.data = master.data

        return loss