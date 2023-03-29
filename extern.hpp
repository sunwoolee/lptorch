void linear_quantize(int cuda_id, at::Tensor data, int bit_num, at::Tensor sca, at::Tensor of, at::Tensor uf, int room);
void linear_quantize_sr(int cuda_id, at::Tensor data, int bit_num, at::Tensor sca, at::Tensor of, at::Tensor uf, int room, at::Tensor rand);
void linear_hysteresis(int cuda_id, at::Tensor pre_data, at::Tensor data, int bit_num, at::Tensor sca, at::Tensor of, at::Tensor uf, int room);

void custom_fp_quantize(int cuda_id, at::Tensor data, at::Tensor man, at::Tensor sca, at::Tensor of, at::Tensor uf, int room);
void custom_fp_quantize_sr(int cuda_id, at::Tensor data, at::Tensor man, at::Tensor sca, at::Tensor of, at::Tensor uf, int room, at::Tensor rand);
void custom_fp_hysteresis(int cuda_id, at::Tensor pre_data, at::Tensor data, at::Tensor man, at::Tensor sca, at::Tensor of, at::Tensor uf, int room);

void fp_quantize(int cuda_id, at::Tensor data, int exp_bit, int man_bit, int bias);
void fp_quantize_sr(int cuda_id, at::Tensor data, int exp_bit, int man_bit, int bias, at::Tensor rand);
void fp_hysteresis(int cuda_id, at::Tensor pre_data, at::Tensor data, int exp_bit, int man_bit, int bias);

void log4_trim_mantissa(int cuda_id, at::Tensor data, at::Tensor sca);