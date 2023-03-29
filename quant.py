import torch
from .major import linear_quantize, custom_fp_quantize, fp_quantize, linear_hysteresis, custom_fp_hysteresis, fp_hysteresis, binary_quantize, binary_hysteresis

class qformat():
    def __init__(self, format_type):
        self._type = format_type

    def get_type(self):
        return self._type

class binary_format(qformat):
    def __init__(self):
        super().__init__('binary')

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
    def __init__(self, qformat, room=0, tracking=True, stochastic=False, ch_wise=False, ch_dim=0):
        """ use accurate scale if tracking is False  """
        self._type = qformat.get_type()
        self.qformat = qformat
        self.room = room
        self.tracking = tracking
        self.stochastic = stochastic
        self.ch_wise = ch_wise
        self.ch_dim = ch_dim

    def quantize(self, input, scale=None, room=None):
        if self.tracking is False:
            scale = None
        if room is None:
            room = self.room
        if self._type == 'linear':
            return linear_quantize(input, scale, self.qformat.bit_num, room, self.stochastic, self.ch_wise, self.ch_dim, self.qformat.unsigned)
        elif self._type == 'custom_fp':
            return custom_fp_quantize(input, scale, self.qformat.man.to(input.device), room, self.stochastic, self.ch_wise, self.ch_dim)
        elif self._type == 'fp':
            return fp_quantize(input, self.qformat.exp_bit, self.qformat.man_bit, self.qformat.bias, self.stochastic)
        elif self._type == 'binary':
            return binary_quantize(input, self.stochastic, self.ch_wise, self.ch_dim)
        
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