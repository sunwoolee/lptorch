import torch
from .major import linear_quantize, custom_fp_quantize, fp_quantize, linear_hysteresis, custom_fp_hysteresis, fp_hysteresis, binary_quantize, binary_hysteresis, ternary_quantize, ternary_hysteresis, linear_block_quantize, linear_ch_block_quantize

class qformat():
    def __init__(self, format_type):
        self._type = format_type

    def get_type(self):
        return self._type

class binary_format(qformat):
    def __init__(self):
        super().__init__('binary')

class ternary_format(qformat):
    def __init__(self):
        super().__init__('ternary')

class linear_block_format(qformat):
    def __init__(self, bit_num, scale_man, scale_log2=False, ):
        super().__init__('linear_block')
        self.bit_num = bit_num
        self.scale_log2 = scale_log2
        self.scale_man = scale_man

class linear_ch_block_format(qformat):
    def __init__(self, bit_num, scale_man, scale_log2=False, ch_scale_bit=None):
        super().__init__('linear_ch_block')
        self.bit_num = bit_num
        self.scale_log2 = scale_log2
        self.scale_man = scale_man
        self.ch_scale_bit = ch_scale_bit

class linear_format(qformat):
    def __init__(self, bit_num, unsigned=False):
        super().__init__('linear')
        if unsigned:
            self.bit_num = bit_num+1
        else:
            self.bit_num = bit_num
        self.scale_diff = bit_num-1
        self.unsigned = unsigned

class custom_fp_format(qformat):
    def __init__(self, man):
        super().__init__('custom_fp')
        self.man = torch.tensor(man, dtype=torch.int)
        self.scale_diff = man[0]

class custom_fp_multi_type_format(qformat):
    def __init__(self, man_list, default_man):
        super().__init__('custom_fp_multi_type')
        self.man = []
        for man in man_list:
            self.man.append(torch.tensor(man, dtype=torch.int))
        self.default_man = torch.tensor(default_man, dtype=torch.int)
        
class fp_format(qformat):
    def __init__(self, exp_bit, man_bit, bias=None):
        super().__init__('fp')
        self.exp_bit = exp_bit
        self.man_bit = man_bit
        if bias is None:
            self.bias = (1 << (exp_bit-1)) - 1
        else:
            self.bias = bias

class quant():
    def __init__(self, qformat, room=0, tracking=True, stochastic=False, ch_wise=False, ch_dim=0, block_wise=False, block_size=0, block_dim=0):
        """ use accurate scale if tracking is False  """
        self._type = qformat.get_type()
        self.qformat = qformat
        self.room = room
        self.tracking = tracking
        self.stochastic = stochastic
        self.ch_wise = ch_wise
        self.ch_dim = ch_dim
        self.block_wise = block_wise
        self.block_size = block_size
        self.block_dim = block_dim
        if isinstance(ch_dim, list):
            tmp = -1
            for ch in ch_dim:
                if tmp >= ch:
                    raise Exception("When ch_dim is list object, elements must be in ascending order")
                tmp = ch

    def quantize(self, input, scale=None, room=None, quant_type=None):
        if self.tracking is False:
            scale = None
        if room is None:
            room = self.room
        if self._type == 'linear':
            return linear_quantize(input, scale, self.qformat.bit_num, room, self.stochastic, self.ch_wise, self.ch_dim, self.block_wise, self.block_size, self.block_dim, self.qformat.unsigned)
        elif self._type == 'custom_fp':
            return custom_fp_quantize(input, scale, self.qformat.man.to(input.device), room, self.stochastic, self.ch_wise, self.ch_dim)
        elif self._type == 'fp':
            return fp_quantize(input, self.qformat.exp_bit, self.qformat.man_bit, self.qformat.bias, self.stochastic)
        elif self._type == 'binary':
            return binary_quantize(input, self.stochastic, self.ch_wise, self.ch_dim)
        elif self._type == 'ternary':
            return ternary_quantize(input, self.stochastic, self.ch_wise, self.ch_dim)
        elif self._type == 'linear_block':
            return linear_block_quantize(input, self.qformat.bit_num, self.block_size, self.block_dim, self.qformat.scale_log2, self.qformat.scale_man)
        elif self._type == 'linear_ch_block':
            return linear_ch_block_quantize(input, self.qformat.bit_num, self.block_size, self.block_dim, self.qformat.scale_log2, self.qformat.scale_man, self.qformat.ch_scale_bit)
        elif self._type == 'custom_fp_multi_type':
            if quant_type == None:
                raise Exception("quant type is None!")
            elif quant_type == -1:
                man = self.qformat.default_man.to(input.device)
            else:
                man = self.qformat.man[quant_type].to(input.device)
            return custom_fp_quantize(input, scale, man, room, self.stochastic, self.ch_wise, self.ch_dim)
        
    def hysteresis(self, pre_input, input, scale=None, room=None):
        if self.tracking is False:
            scale = None
        if room is None:
            room = self.room
        if self._type == 'linear':
            return linear_hysteresis(pre_input, input, scale, self.qformat.bit_num, room, self.ch_wise, self.ch_dim, self.qformat.unsigned)
        elif self._type == 'custom_fp':
            return custom_fp_hysteresis(pre_input, input, scale, self.qformat.man.to(input.device), room, self.ch_wise, self.ch_dim)
        elif self._type == 'fp':
            return fp_hysteresis(pre_input, input, self.qformat.exp_bit, self.qformat.man_bit, self.qformat.bias)
        elif self._type == 'binary':
            return binary_hysteresis(pre_input, input, self.ch_wise, self.ch_dim)
        elif self._type == 'ternary':
            return ternary_hysteresis(pre_input, input, self.ch_wise, self.ch_dim)
        elif self._type == 'linear_block':
            raise Exception("hysteresis for linear_block is not implemented.")
