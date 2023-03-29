#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include "lptorch.cuh"
#include "utils.hpp"

void linear_quantize(int cuda_id, at::Tensor data, int bit_num, at::Tensor sca, at::Tensor of, at::Tensor uf, int room) {
  int N = data.numel();
  int EXCH = N / sca.numel();
  linear_quantize_kernel(cuda_id, data.data_ptr<float>(), bit_num, sca.data_ptr<int>(), of.data_ptr<bool>(), uf.data_ptr<bool>(), room, 
    N, EXCH, at::cuda::getCurrentCUDAStream());
}
void linear_quantize_sr(int cuda_id, at::Tensor data, int bit_num, at::Tensor sca, at::Tensor of, at::Tensor uf, int room, at::Tensor rand) {
  int N = data.numel();
  int EXCH = N / sca.numel();
  linear_quantize_sr_kernel(cuda_id, data.data_ptr<float>(), bit_num, sca.data_ptr<int>(), of.data_ptr<bool>(), uf.data_ptr<bool>(), room, rand.data_ptr<float>(), 
    N, EXCH, at::cuda::getCurrentCUDAStream());
}
void linear_hysteresis(int cuda_id, at::Tensor pre_data, at::Tensor data, int bit_num, at::Tensor sca, at::Tensor of, at::Tensor uf, int room) {
  int N = data.numel();
  int EXCH = N / sca.numel();
  linear_hysteresis_kernel(cuda_id, pre_data.data_ptr<float>(), data.data_ptr<float>(), bit_num, sca.data_ptr<int>(), of.data_ptr<bool>(), uf.data_ptr<bool>(), room, 
    N, EXCH, at::cuda::getCurrentCUDAStream());
}

void custom_fp_quantize(int cuda_id, at::Tensor data, at::Tensor man, at::Tensor sca, at::Tensor of, at::Tensor uf, int room) {
  int N = data.numel();
  int MN = man.numel();
  int EXCH = N / sca.numel();
  custom_fp_quantize_kernel(cuda_id, data.data_ptr<float>(), man.data_ptr<int>(), sca.data_ptr<int>(), of.data_ptr<bool>(), uf.data_ptr<bool>(), room,
    N, MN, EXCH, at::cuda::getCurrentCUDAStream());
}
void custom_fp_quantize_sr(int cuda_id, at::Tensor data, at::Tensor man, at::Tensor sca, at::Tensor of, at::Tensor uf, int room, at::Tensor rand) {
  int N = data.numel();
  int MN = man.numel();
  int EXCH = N / sca.numel();
  custom_fp_quantize_sr_kernel(cuda_id, data.data_ptr<float>(), man.data_ptr<int>(), sca.data_ptr<int>(), of.data_ptr<bool>(), uf.data_ptr<bool>(), room, rand.data_ptr<float>(),
    N, MN, EXCH, at::cuda::getCurrentCUDAStream());
}
void custom_fp_hysteresis(int cuda_id, at::Tensor pre_data, at::Tensor data, at::Tensor man, at::Tensor sca, at::Tensor of, at::Tensor uf, int room) {
  int N = data.numel();
  int MN = man.numel();
  int EXCH = N / sca.numel();
  custom_fp_hysteresis_kernel(cuda_id, pre_data.data_ptr<float>(), data.data_ptr<float>(), man.data_ptr<int>(), sca.data_ptr<int>(), of.data_ptr<bool>(), uf.data_ptr<bool>(), room,
    N, MN, EXCH, at::cuda::getCurrentCUDAStream());
}

void fp_quantize(int cuda_id, at::Tensor data, int exp_bit, int man_bit, int bias) {
  int N = data.numel();
  fp_quantize_kernel(cuda_id, data.data_ptr<float>(), exp_bit, man_bit, bias, N, at::cuda::getCurrentCUDAStream());
}
void fp_quantize_sr(int cuda_id, at::Tensor data, int exp_bit, int man_bit, int bias, at::Tensor rand) {
  int N = data.numel();
  fp_quantize_sr_kernel(cuda_id, data.data_ptr<float>(), exp_bit, man_bit, bias, rand.data_ptr<float>(), N, at::cuda::getCurrentCUDAStream());
}
void fp_hysteresis(int cuda_id, at::Tensor pre_data, at::Tensor data, int exp_bit, int man_bit, int bias) {
  int N = data.numel();
  fp_hysteresis_kernel(cuda_id, pre_data.data_ptr<float>(), data.data_ptr<float>(), exp_bit, man_bit, bias, N, at::cuda::getCurrentCUDAStream());
}

void log4_trim_mantissa(int cuda_id, at::Tensor data, at::Tensor sca) {
  int N = data.numel();
  log4_trim_mantissa_kernel(cuda_id, data.data_ptr<float>(), sca.data_ptr<int>(), N, at::cuda::getCurrentCUDAStream());
}

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