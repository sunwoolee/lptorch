import torch
import os
import lptorch_cuda

error_quant = None
activ_quant = None
shortcut_quant = None
shortcut_error_quant = None
weight_quant = None
bias_quant = None
grad_quant = None
master_quant = None

hysteresis_update = False
track_distribution = False

torch.manual_seed(0)

fine_tuning_step = 2
quant_buffer = {}

def quant_update():
	global fine_tuning_step, quant_buffer
	global error_quant, activ_quant, shortcut_quant, shortcut_error_quant, weight_quant, bias_quant, grad_quant, master_quant
	if fine_tuning_step == 0:		# full_precision_training
		if error_quant is not None:	
			quant_buffer['error'] = error_quant
			error_quant = None
		if activ_quant is not None:	
			quant_buffer['activ'] = activ_quant
			activ_quant = None
		if shortcut_quant is not None:	
			quant_buffer['shortcut'] = shortcut_quant
			shortcut_quant = None
		if shortcut_error_quant is not None:	
			quant_buffer['shortcut_error'] = shortcut_error_quant
			shortcut_error_quant = None
		if weight_quant is not None:	
			quant_buffer['weight'] = weight_quant
			weight_quant = None
		if bias_quant is not None:	
			quant_buffer['bias'] = bias_quant
			bias_quant = None
		if grad_quant is not None:	
			quant_buffer['grad'] = grad_quant
			grad_quant = None
		if master_quant is not None:	
			quant_buffer['master'] = master_quant
			master_quant = None
	elif fine_tuning_step == 1:	# weight quantization
		if 'weight' in quant_buffer.keys():
			weight_quant = quant_buffer['weight']
			del quant_buffer['weight']
		if 'bias' in quant_buffer.keys():
			bias_quant = quant_buffer['bias']
			del quant_buffer['bias']
		if 'grad' in quant_buffer.keys():
			grad_quant = quant_buffer['grad']
			del quant_buffer['grad']
		if 'master' in quant_buffer.keys():
			master_quant = quant_buffer['master']
			del quant_buffer['master']
	else:							# activation quantization
		if 'error' in quant_buffer.keys():
			error_quant = quant_buffer['error']
			del quant_buffer['error']
		if 'activ' in quant_buffer.keys():
			activ_quant = quant_buffer['activ']
			del quant_buffer['activ']
		if 'shortcut' in quant_buffer.keys():
			shortcut_quant = quant_buffer['shortcut']
			del quant_buffer['shortcut']
		if 'shortcut_error' in quant_buffer.keys():
			shortcut_error_quant = quant_buffer['shortcut_error']
			del quant_buffer['shortcut_error']

def set_fine_tuning_step(step):
	global fine_tuning_step
	fine_tuning_step = step
	quant_update()

def step_fine_tuning():
	global fine_tuning_step
	fine_tuning_step += 1
	quant_update()

def get_fine_tuning_step():
	return fine_tuning_step

def get_error_quant():
	return error_quant

def set_error_quant(value):
	global error_quant
	from .quant import quant
	if type(value) is quant:
		error_quant = value
	else:
		error_quant = None
		print('type of quant must be quant.quant')
	quant_update()

def get_activ_quant():
	return activ_quant

def set_activ_quant(value):
	global activ_quant
	from .quant import quant
	if type(value) is quant:
		activ_quant = value
	else:
		activ_quant = None
		print('type of quant must be quant.quant')
	quant_update()

def get_shortcut_quant():
	return shortcut_quant

def set_shortcut_quant(value):
	global shortcut_quant
	from .quant import quant
	if type(value) is quant:
		shortcut_quant = value
	else:
		shortcut_quant = None
		print('type of quant must be quant.quant')
	quant_update()

def get_shortcut_error_quant():
	return shortcut_error_quant

def set_shortcut_error_quant(value):
	global shortcut_error_quant
	from .quant import quant
	if type(value) is quant:
		shortcut_error_quant = value
	else:
		shortcut_error_quant = None
		print('type of quant must be quant.quant')
	quant_update()

def get_weight_quant():
	return weight_quant

def set_weight_quant(value):
	global weight_quant
	from .quant import quant
	if type(value) is quant:
		weight_quant = value
	else:
		weight_quant = None
		print('type of quant must be quant.quant')
	quant_update()

def get_bias_quant():
	return bias_quant

def set_bias_quant(value):
	global bias_quant
	from .quant import quant
	if type(value) is quant:
		bias_quant = value
	else:
		bias_quant = None
		print('type of quant must be quant.quant')
	quant_update()

def get_grad_quant():
	return grad_quant

def set_grad_quant(value):
	global grad_quant
	from .quant import quant
	if type(value) is quant:
		grad_quant = value
	else:
		grad_quant = None
		print('type of quant must be quant.quant')
	quant_update()

def get_master_quant():
	return master_quant

def set_master_quant(value):
	global master_quant
	from .quant import quant
	if type(value) is quant:
		master_quant = value
	else:
		master_quant = None
		print('type of quant must be quant.quant')
	quant_update()

def get_hysteresis_update():
	return hysteresis_update

def set_hysteresis_update(value):
	global hysteresis_update
	if type(value) is bool:
		hysteresis_update = value
	else:
		print('type of hysteresis_update must be bool')

def set_track_distribution(value):
	global track_distribution
	if type(value) is bool:
		track_distribution = value
	else:
		print('type of track_distribution must be bool')

def ch_total(tensor, ch_dim):
    dim = 1
    if isinstance(ch_dim, list):
        for ch in ch_dim:
            dim *= tensor.shape[ch]
    else:
        dim = tensor.shape[ch_dim]
    return dim

def ch_swapaxes(tensor, ch_dim):
    if isinstance(ch_dim, list):
        for idx, ch in enumerate(ch_dim):
            tensor = tensor.swapaxes(idx, ch)
    else:
        tensor = tensor.swapaxes(0, ch_dim)
    return tensor

def ch_recovery(tensor, ch_dim):
    if isinstance(ch_dim, list):
        for idx, ch in enumerate(reversed(ch_dim)):
            tensor = tensor.swapaxes(len(ch_dim)-1-idx, ch)
    else:
        tensor = tensor.swapaxes(0, ch_dim)
    return tensor

def block_reshape(tensor, block_size, block_dim):
	tensor = tensor.swapaxes(block_dim, -1)
	swapped_shape = tensor.shape
	if tensor.shape[-1]%block_size != 0:
		zero = torch.zeros(list(tensor.shape)[:-1]+[block_size-tensor.shape[-1]%block_size]).to(tensor.device)
		tensor = torch.concat([tensor, zero], -1)
	tensor = tensor.reshape(-1, block_size)
	return tensor, swapped_shape

def block_recovery(tensor, swapped_shape, block_size, block_dim):
	if swapped_shape[-1]%block_size != 0:
		tensor = tensor.reshape(-1, swapped_shape[-1]-swapped_shape[-1]%block_size+block_size)[:,:swapped_shape[-1]]
	tensor = tensor.reshape(swapped_shape)
	tensor = tensor.swapaxes(block_dim, -1)
	return tensor

def block_scale_num(tensor, block_size, block_dim):
	total_num = 1
	for idx, dim in enumerate(tensor.shape):
		if idx == block_dim:
			block_dim_size = dim
		else:
			total_num *= dim
	if block_dim_size%block_size != 0:
		block_dim_size += block_size - block_dim_size%block_size
	return total_num*block_dim_size

def binary_quantize(tensor, stochastic, ch_wise, ch_dim):
    if ch_wise:
        scale = ch_swapaxes(tensor.abs(), ch_dim).reshape(ch_total(tensor, ch_dim),-1).mean(1)
    else:
        scale = tensor.abs().mean()
    if stochastic:
        rand = torch.rand_like(scale).add(scale.log2())
        scale = torch.where(scale>0, rand.floor(), scale).int()
    else:
        scale = torch.where(scale>0, scale.log2().round(), scale).int()
    tensor = torch.where(tensor>0, tensor.mul(0).add(1), tensor.mul(0).add(-1))
    if ch_wise:
        tensor = ch_swapaxes(tensor, ch_dim)
        shape = tensor.shape
        tensor = tensor.reshape(ch_total(tensor, ch_dim), -1).swapaxes(0,1).mul(torch.pow(2,scale.float())).swapaxes(0,1).reshape(shape)
        tensor = ch_recovery(tensor, ch_dim)
    else:
        tensor = tensor.mul(torch.pow(2,scale.float()))
    tensor.scale = scale
    return tensor

def binary_hysteresis(pre_tensor, tensor, ch_wise, ch_dim):
	if ch_wise:
		pre_scale = ch_swapaxes(pre_tensor.abs(), ch_dim).reshape(ch_total(pre_tensor, ch_dim),-1).max(1)[0]
		scale = ch_swapaxes(tensor.abs(), ch_dim).reshape(ch_total(tensor, ch_dim),-1).mean(1)
	else:
		pre_scale = pre_tensor.abs().max()
		scale = tensor.abs().mean()
	pre_scale = torch.where(pre_scale>0, pre_scale.log2(), pre_scale)
	scale = torch.where(scale>0, scale.log2(), scale)
	scale = torch.where(scale > pre_scale, scale.floor(), scale.ceil()).int()
	tensor = torch.where(tensor>0, tensor.mul(0).add(1), tensor.mul(0).add(-1))
	if ch_wise:
		tensor = ch_swapaxes(tensor, ch_dim)
		shape = tensor.shape
		tensor = tensor.reshape(ch_total(tensor, ch_dim), -1).swapaxes(0,1).mul(torch.pow(2,scale.float())).swapaxes(0,1).reshape(shape)
		tensor = ch_recovery(tensor, ch_dim)
	else:
		tensor = tensor.mul(torch.pow(2,scale.float()))
	tensor.scale = scale
	return tensor

def ternary_quantize(tensor, stochastic, ch_wise, ch_dim):
	if ch_wise:
		scale = ch_swapaxes(tensor.abs(), ch_dim).reshape(ch_total(tensor, ch_dim),-1).mean(1)
	else:
		scale = tensor.abs().mean()
	if stochastic:
		rand = torch.rand_like(scale).add(scale.log2())
		scale = torch.where(scale>0, rand.floor(), scale).int()
	else:
		scale = torch.where(scale>0, scale.log2().round(), scale).int()
	tensor_2 = tensor.mul(2)
	tensor = torch.where(tensor>0, tensor.mul(0).add(1), tensor.mul(0).add(-1))
	if ch_wise:
		tensor = ch_swapaxes(tensor, ch_dim)
		shape = tensor.shape
		tensor = tensor.reshape(ch_total(tensor, ch_dim), -1).swapaxes(0,1).mul(torch.pow(2,scale.float())).swapaxes(0,1).reshape(shape)
		tensor = ch_recovery(tensor, ch_dim)
	else:
		tensor = tensor.mul(torch.pow(2,scale.float()))
	tensor = torch.where(tensor.abs()>tensor_2.abs(), tensor.mul(0), tensor)
	tensor.scale = scale
	return tensor

def ternary_hysteresis(pre_tensor, tensor, ch_wise, ch_dim):
	if ch_wise:
		pre_scale = ch_swapaxes(pre_tensor.abs(), ch_dim).reshape(ch_total(pre_tensor, ch_dim),-1).max(1)[0]
		scale = ch_swapaxes(tensor.abs(), ch_dim).reshape(ch_total(tensor, ch_dim),-1).mean(1)
	else:
		pre_scale = pre_tensor.abs().max()
		scale = tensor.abs().mean()
	pre_scale = torch.where(pre_scale>0, pre_scale.log2(), pre_scale)
	scale = torch.where(scale>0, scale.log2(), scale)
	scale = torch.where(scale > pre_scale, scale.floor(), scale.ceil()).int()
	tensor_2 = tensor.mul(2)
	tensor = torch.where(tensor>0, tensor.mul(0).add(1), tensor.mul(0).add(-1))
	if ch_wise:
		tensor = ch_swapaxes(tensor, ch_dim)
		shape = tensor.shape
		tensor = tensor.reshape(ch_total(tensor, ch_dim), -1).swapaxes(0,1).mul(torch.pow(2,scale.float())).swapaxes(0,1).reshape(shape)
		tensor = ch_recovery(tensor, ch_dim)
	else:
		tensor = tensor.mul(torch.pow(2,scale.float()))
	tensor = torch.where(tensor.abs()>tensor_2.abs(), tensor.mul(0), tensor)
	tensor.scale = scale
	return tensor

def linear_quantize(tensor, scale, bit_num, room, stochastic, ch_wise, ch_dim, block_wise, block_size, block_dim, unsigned):
	if unsigned:
		tensor = torch.where(tensor<0, tensor.mul(0), tensor)
	if scale is None:
		if ch_wise:
			scale = ch_swapaxes(tensor.abs(), ch_dim).reshape(ch_total(tensor, ch_dim),-1).max(1)[0]
		elif block_wise:
			scale = block_reshape(tensor.abs(), block_size, block_dim)[0].max(1)[0]
		else:
			scale = tensor.abs().max()
		scale = torch.where(scale>0, scale.log2().floor(), scale)
		scale = scale.int().add(room)
	else:
		if len(scale.shape) == 0:
			scale.data = scale.data.unsqueeze(0)
		if ch_wise == True and scale.shape[0] != tensor.shape[ch_dim]:
			scale.data = scale.repeat(tensor.shape[ch_dim]).data
		elif block_wise == True and scale.shape[0] != block_scale_num(tensor, block_size, block_dim):
			scale.data = scale.repeat(block_scale_num(tensor, block_size, block_dim)).data

	cuda_id = tensor.get_device()
	if ch_wise:
		tensor = ch_swapaxes(tensor, ch_dim)
	elif block_wise:
		tensor, swapped_shape = block_reshape(tensor, block_size, block_dim)
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	overflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	underflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	if stochastic:
		rand = torch.rand_like(tensor).contiguous()
		lptorch_cuda.linear_quantize_sr(cuda_id, tensor, bit_num, scale, overflow, underflow, room, rand)
	else:
		lptorch_cuda.linear_quantize(cuda_id, tensor, bit_num, scale, overflow, underflow, room)
	tensor = tensor.reshape(shape)
	if ch_wise:
		tensor = ch_recovery(tensor, ch_dim)
		overflow = overflow.reshape(ch_total(tensor, ch_dim),-1).max(1)[0].int()
		underflow = underflow.reshape(ch_total(tensor, ch_dim),-1).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	elif block_wise:
		tensor = block_recovery(tensor, swapped_shape, block_size, block_dim)
		overflow = overflow.reshape(-1, block_size).max(1)[0].int()
		underflow = underflow.reshape(-1, block_size).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	else:
		if overflow.max() == 1: scale.add_(1)
		elif underflow.max() == 0: scale.add_(-1)
	tensor.scale = scale.clone()
	return tensor

def linear_block_quantize(tensor, bit_num, block_size, block_dim, scale_log2, scale_man):
	cuda_id = tensor.get_device()
	tensor, swapped_shape = block_reshape(tensor, block_size, block_dim)
	max_val = tensor.max(1)[0]
	min_val = tensor.min(1)[0]
	abs_max_val = torch.where(max_val.abs() > min_val.abs(), max_val.abs(), min_val.abs())
	if scale_log2:
		scale = torch.pow(2,torch.where(abs_max_val>0, abs_max_val.log2().floor().add(-bit_num+2), abs_max_val))
	else:
		scale = torch.where(max_val == abs_max_val, max_val.div(2**(bit_num-1)-1), min_val.abs().div(2**(bit_num-1)))
		man = torch.tensor(scale_man, dtype=torch.int).to(scale.device)
		overflow = torch.empty(scale.size(), dtype=torch.bool, device=scale.device)
		underflow = torch.empty(scale.size(), dtype=torch.bool, device=scale.device)
		s_scale = scale.abs().max()
		s_scale = torch.where(s_scale>0, s_scale.log2().floor(), s_scale).int()
		lptorch_cuda.custom_fp_quantize(cuda_id, scale, man, s_scale, overflow, underflow, 0)
		# lptorch_cuda.fp_quantize(cuda_id, scale, 8, scale_bit_num-1, 127)
	scale = torch.where(scale == 0, scale.add(1), scale)
	scale = scale.unsqueeze(1)
	tensor = tensor.div(scale).round()
	tensor = torch.where(tensor > 2**(bit_num-1)-1, tensor.mul(0).add(2**(bit_num-1)-1), tensor)
	tensor = torch.where(tensor < -2**(bit_num-1), tensor.mul(0).add(-2**(bit_num-1)), tensor)
	tensor = tensor.mul(scale)
	tensor = block_recovery(tensor, swapped_shape, block_size, block_dim)
	tensor.scale = scale
	return tensor

def linear_ch_block_quantize(tensor, bit_num, block_size, block_dim, scale_log2, scale_man, ch_scale_bit):
	cuda_id = tensor.get_device()
	ch_scale = tensor.abs().max(dim=block_dim, keepdim=True)[0]
	ch_scale = torch.where(ch_scale>0, ch_scale.log2().floor(), ch_scale)
	if ch_scale_bit is not None:
		ch_scale_max = ch_scale.max()
		ch_scale_min = ch_scale_max.add(2-(2**ch_scale_bit))
		ch_scale = torch.where(ch_scale>ch_scale_min, ch_scale, ch_scale_min)
	ch_scale = torch.pow(2,ch_scale)
	tensor = tensor.div(ch_scale)
	tensor, swapped_shape = block_reshape(tensor, block_size, block_dim)
	max_val = tensor.max(1)[0]
	min_val = tensor.min(1)[0]
	abs_max_val = torch.where(max_val.abs() > min_val.abs(), max_val.abs(), min_val.abs())
	if scale_log2:
		scale = torch.pow(2,torch.where(abs_max_val>0, abs_max_val.log2().floor().add(-bit_num+2), abs_max_val))
	else:
		scale = torch.where(max_val == abs_max_val, max_val.div(2**(bit_num-1)-1), min_val.abs().div(2**(bit_num-1)))
		man = torch.tensor(scale_man, dtype=torch.int).to(scale.device)
		overflow = torch.empty(scale.size(), dtype=torch.bool, device=scale.device)
		underflow = torch.empty(scale.size(), dtype=torch.bool, device=scale.device)
		s_scale = scale.abs().max()
		s_scale = torch.where(s_scale>0, s_scale.log2().floor(), s_scale).int()
		s_scale_min_val = torch.pow(2,s_scale.float().add(2-len(scale_man)))
		lptorch_cuda.custom_fp_quantize(cuda_id, scale, man, s_scale, overflow, underflow, 0)
		# lptorch_cuda.fp_quantize(cuda_id, scale, 8, scale_bit_num-1, 127)
	scale = torch.where(scale == 0, scale.add(s_scale_min_val), scale)
	scale = scale.unsqueeze(1)
	tensor = tensor.div(scale).round()
	tensor = torch.where(tensor > 2**(bit_num-1)-1, tensor.mul(0).add(2**(bit_num-1)-1), tensor)
	tensor = torch.where(tensor < -2**(bit_num-1), tensor.mul(0).add(-2**(bit_num-1)), tensor)
	tensor = tensor.mul(scale)
	tensor = block_recovery(tensor, swapped_shape, block_size, block_dim)

	tensor = tensor.mul(ch_scale)
	tensor.scale = scale
	tensor.ch_scale = ch_scale
	return tensor

def linear_hysteresis(pre_tensor, tensor, scale, bit_num, room, ch_wise, ch_dim, unsigned):
	if unsigned:
		tensor = torch.where(tensor<0, tensor.mul(0), tensor)
	if scale is None:
		if ch_wise:
			scale = ch_swapaxes(tensor.abs(), ch_dim).reshape(ch_total(tensor, ch_dim),-1).max(1)[0]
		else:
			scale = tensor.abs().max()
		scale = torch.where(scale>0, scale.log2().floor(), scale)
		scale = scale.int().add(room)
	else:
		if len(scale.shape) == 0:
			scale.data = scale.data.unsqueeze(0)
		if ch_wise == True and scale.shape[0] != tensor.shape[ch_dim]:
			scale.data = scale.repeat(tensor.shape[ch_dim]).data
	cuda_id = tensor.get_device()
	if ch_wise:
		tensor = ch_swapaxes(tensor, ch_dim)
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	pre_tensor = pre_tensor.reshape(-1)
	overflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	underflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	lptorch_cuda.linear_hysteresis(cuda_id, pre_tensor, tensor, bit_num, scale, overflow, underflow, room)
	tensor = tensor.reshape(shape)
	if ch_wise:
		tensor = ch_recovery(tensor, ch_dim)
		overflow = overflow.reshape(ch_total(tensor, ch_dim),-1).max(1)[0].int()
		underflow = underflow.reshape(ch_total(tensor, ch_dim),-1).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	else:
		if overflow.max() == 1: scale.add_(1)
		elif underflow.max() == 0: scale.add_(-1)
	tensor.scale = scale.clone()
	return tensor

def custom_fp_quantize(tensor, scale, man, room, stochastic, ch_wise, ch_dim):
	if scale is None:
		if ch_wise:
			scale = ch_swapaxes(tensor.abs(), ch_dim).reshape(ch_total(tensor, ch_dim),-1).max(1)[0]
		else:
			scale = tensor.abs().max()
		scale = torch.where(scale>0, scale.log2().floor(), scale)
		scale = scale.int().add(room)
	else:
		if len(scale.shape) == 0:
			scale.data = scale.data.unsqueeze(0)
		if ch_wise == True and scale.shape[0] != tensor.shape[ch_dim]:
			scale.data = scale.repeat(tensor.shape[ch_dim]).data
	cuda_id = tensor.get_device()
	if ch_wise:
		tensor = ch_swapaxes(tensor, ch_dim)
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	overflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	underflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	if stochastic:
		rand = torch.rand_like(tensor).contiguous()
		lptorch_cuda.custom_fp_quantize_sr(cuda_id, tensor, man, scale, overflow, underflow, room, rand)
	else:
		lptorch_cuda.custom_fp_quantize(cuda_id, tensor, man, scale, overflow, underflow, room)
	tensor = tensor.reshape(shape)
	if ch_wise:
		tensor = ch_recovery(tensor, ch_dim)
		overflow = overflow.reshape(ch_total(tensor, ch_dim),-1).max(1)[0].int()
		underflow = underflow.reshape(ch_total(tensor, ch_dim),-1).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	else:
		if overflow.max() == 1: scale.add_(1)
		elif underflow.max() == 0: scale.add_(-1)
	tensor.scale = scale.clone()
	return tensor

def custom_fp_hysteresis(pre_tensor, tensor, scale, man, room, ch_wise, ch_dim):
	if scale is None:
		if ch_wise:
			scale = ch_swapaxes(tensor.abs(), ch_dim).reshape(ch_total(tensor, ch_dim),-1).max(1)[0]
		else:
			scale = tensor.abs().max()
		scale = torch.where(scale>0, scale.log2().floor(), scale)
		scale = scale.int().add(room)
	else:
		if len(scale.shape) == 0:
			scale.data = scale.data.unsqueeze(0)
		if ch_wise == True and scale.shape[0] != tensor.shape[ch_dim]:
			scale.data = scale.repeat(tensor.shape[ch_dim]).data
	cuda_id = tensor.get_device()
	if ch_wise:
		tensor = ch_swapaxes(tensor, ch_dim)
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	pre_tensor = pre_tensor.reshape(-1)
	overflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	underflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	lptorch_cuda.custom_fp_hysteresis(cuda_id, pre_tensor, tensor, man, scale, overflow, underflow, room)
	tensor = tensor.reshape(shape)
	if ch_wise:
		tensor = ch_recovery(tensor, ch_dim)
		overflow = overflow.reshape(ch_total(tensor, ch_dim),-1).max(1)[0].int()
		underflow = underflow.reshape(ch_total(tensor, ch_dim),-1).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	else:
		if overflow.max() == 1: scale.add_(1)
		elif underflow.max() == 0: scale.add_(-1)
	tensor.scale = scale.clone()
	return tensor

def fp_quantize(tensor, exp_bit, man_bit, bias, stochastic):
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	if stochastic:
		rand = torch.rand_like(tensor).contiguous()
		lptorch_cuda.fp_quantize_sr(cuda_id, tensor, exp_bit, man_bit, bias, rand)
	else:
		lptorch_cuda.fp_quantize(cuda_id, tensor, exp_bit, man_bit, bias)
	tensor = tensor.reshape(shape)
	tensor.scale = torch.zeros(1).int().to(tensor.device)
	return tensor

def fp_hysteresis(pre_tensor, tensor, exp_bit, man_bit, bias):
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	pre_tensor = pre_tensor.reshape(-1)
	lptorch_cuda.fp_hysteresis(cuda_id, pre_tensor, tensor, exp_bit, man_bit, bias)
	tensor = tensor.reshape(shape)
	tensor.scale = torch.zeros(1).int().to(tensor.device)
	return tensor

def log4_trim_mantissa(tensor, scale):
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.clone().reshape(-1)
	lptorch_cuda.log4_trim_mantissa(cuda_id, tensor, scale)
	tensor = tensor.reshape(shape)
	tensor.scale = scale
	return tensor

def load_state_dict(target_model, state, merged_bn=False):
	t_state = target_model.state_dict()
	
	saved_key = state.keys()
	target_key = t_state.keys()
	missing_key = list(set(target_key) - set(saved_key))
	unexpected_key = list(set(saved_key) - set(target_key))
	
	missing_wo = []
	unexpected_wo = []
	for i in range(len(missing_key)):
		missing = missing_key[i].replace('lp_module.', '')
		missing_wo.append(missing)
	for i in range(len(unexpected_key)):
		unexpected = unexpected_key[i].replace('lp_module.', '')
		unexpected_wo.append(unexpected)
	for j in range(len(unexpected_wo)):
		for i in range(len(missing_wo)):
			if unexpected_wo[j] == missing_wo[i]:
				state[missing_key[i]] = state.pop(unexpected_key[j])
	
	state = {k: v for k, v in state.items() if k in t_state}
	t_state.update(state)
	target_model.load_state_dict(t_state)

	if merged_bn:
		merged_bn = []
		# from model_wrapper import Quantized_Model
		# if not isinstance(target_model, Quantized_Model):
		# 	raise Exception("model is not instance of lptorch.model.Quantized_Model!")
		for bn in target_model.lp_module.modules():
			if isinstance(bn, torch.nn.BatchNorm2d) or isinstance(bn, torch.nn.BatchNorm1d):
				bn.weight.requires_grad = False
				bn.bias.requires_grad = False
				bn.eps = 0
				bn.eval()
				merged_bn.append(bn)
		target_model.merged_bn = merged_bn
		target_model.bn_merge = True

def get_distribution(data, distribution):
	abs_data = data.abs()
	if 'zero' in distribution.keys():
		distribution['zero'] += torch.sum(abs_data==0)
	else:
		distribution['zero'] = torch.sum(abs_data==0)
	scale = abs_data[abs_data>0].log2().int()
	if scale.numel() == 0:
		return
	min_val = scale.min().item()
	max_val = scale.max().item()
	for i in range(min_val, max_val+1):
		if str(i) in distribution.keys():
			distribution[str(i)] += torch.sum(scale==i)
		else:
			distribution[str(i)] = torch.sum(scale==i)
