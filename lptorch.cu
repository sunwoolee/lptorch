#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "gpu.cuh"
#include "utils.hpp"
#include <cuda_fp16.h>
#include <stdio.h>
//#include <cuda_runtime.h>
// Kernel function to add the elements of two arrays

__device__ float stochastic_round(float data, float rand){
  float output;
  output = floorf(data);
  if (output+rand < data) output = output + 1;
  return output;
}
__global__ void linear_quantize_op(int n, int exch, float *data, int bit_num, int *scale, bool *overflow, bool *underflow, int room){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float temp, max, min, of_num, uf_num;
  max = (1 << (bit_num-1)) - 1;
  min = - (1 << (bit_num-1)); 
  of_num = 1 << (bit_num-1-room);
  uf_num = 1 << (bit_num-1-room-1);
  int shift_num;
  for (int i = index; i < n; i += stride) {
    shift_num = bit_num-2-scale[int(i/exch)];
    overflow[i] = 0;
    underflow[i] = 0;
    temp = rintf(scalbnf(data[i], shift_num));
    if (fabsf(temp) >= of_num) overflow[i] = 1;
    if (fabsf(temp) >= uf_num) underflow[i] = 1;
    if (temp > max) temp = max;
    if (temp < min) temp = min;
    data[i] = scalbnf(temp, -shift_num);
  }
}
__global__ void linear_quantize_sr_op(int n, int exch, float *data, int bit_num, int *scale, bool *overflow, bool *underflow, int room, float *rand){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float temp, max, min, of_num, uf_num;
  max = (1 << (bit_num-1)) - 1;
  min = - (1 << (bit_num-1)); 
  of_num = 1 << (bit_num-1-room);
  uf_num = 1 << (bit_num-1-room-1);
  int shift_num;
  for (int i = index; i < n; i += stride) {
    shift_num = bit_num-2-scale[int(i/exch)];
    overflow[i] = 0;
    underflow[i] = 0;
    temp = stochastic_round(scalbnf(data[i], shift_num), rand[i]);
    if (fabsf(temp) >= of_num) overflow[i] = 1;
    if (fabsf(temp) >= uf_num) underflow[i] = 1;
    if (temp > max) temp = max;
    if (temp < min) temp = min;
    data[i] = scalbnf(temp, -shift_num);
  }
}
__global__ void linear_hysteresis_op(int n, int exch, float *pre_data, float *data, int bit_num, int *scale, bool *overflow, bool *underflow, int room){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float temp, max, min, of_num, uf_num;
  max = (1 << (bit_num-1)) - 1;
  min = - (1 << (bit_num-1)); 
  of_num = 1 << (bit_num-1-room);
  uf_num = 1 << (bit_num-1-room-1);
  int shift_num;
  for (int i = index; i < n; i += stride) {
    shift_num = bit_num-2-scale[int(i/exch)];
    overflow[i] = 0;
    underflow[i] = 0;
    temp = scalbnf(data[i], shift_num);
    if (pre_data[i] > data[i]) temp = ceilf(temp);
    else                       temp = floorf(temp);
    if (fabsf(temp) >= of_num) overflow[i] = 1;
    if (fabsf(temp) >= uf_num) underflow[i] = 1;
    if (temp > max) temp = max;
    if (temp < min) temp = min;
    data[i] = scalbnf(temp, -shift_num);
  }
}

__global__ void custom_fp_quantize_op(int n, int mn, int exch, float *data, int *man, int *scale, bool *overflow, bool *underflow, int room){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int expo, diff;
  float temp, max;
  int sca;
  for (int i = index; i < n; i += stride) {
    sca = scale[int(i/exch)];
    max = scalbnf(1,sca + 1) - scalbnf(1,sca+1-man[0]);
    overflow[i] = 0;
    underflow[i] = 0;
    temp = frexpf(data[i], &expo);
    diff = sca - expo + 1;
    if (mn <= diff) {
      temp = scalbnf(temp, mn-1-diff);
      expo = sca - mn + 2;
      diff = mn-1;
    }
    if (0 <= diff) temp = scalbnf(rintf(scalbnf(temp, man[diff])), -man[diff]);
    temp = scalbnf(temp, expo);
    if (temp != 0) {
      frexpf(temp, &expo);
      diff = sca - expo + 1;
      if (diff < room) overflow[i] = 1;
      if (diff < room + 1) underflow[i] = 1;
      if (temp > max) temp = max;
      if (temp < -max) temp = -max;
    }
    data[i] = temp;
  }
}
__global__ void custom_fp_quantize_sr_op(int n, int mn, int exch, float *data, int *man, int *scale, bool *overflow, bool *underflow, int room, float *rand){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int expo, diff;
  float temp, max;
  int sca;
  for (int i = index; i < n; i += stride) {
    sca = scale[int(i/exch)];
    max = scalbnf(1,sca + 1) - scalbnf(1,sca+1-man[0]);
    overflow[i] = 0;
    underflow[i] = 0;
    temp = frexpf(data[i], &expo);
    diff = sca - expo + 1;
    if (mn <= diff) {
      temp = scalbnf(temp, mn-1-diff);
      expo = sca - mn + 2;
      diff = mn-1;
    }
    if (0 <= diff) temp = scalbnf(stochastic_round(scalbnf(temp, man[diff]), rand[i]), -man[diff]);
    temp = scalbnf(temp, expo);
    if (temp != 0) {
      frexpf(temp, &expo);
      diff = sca - expo + 1;
      if (diff < room) overflow[i] = 1;
      if (diff < room + 1) underflow[i] = 1;
      if (temp > max) temp = max;
      if (temp < -max) temp = -max;
    }
    data[i] = temp;
  }
}
__global__ void custom_fp_hysteresis_op(int n, int mn, int exch, float *pre_data, float *data, int *man, int *scale, bool *overflow, bool *underflow, int room){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int expo, diff;
  float temp, max;
  int sca;
  for (int i = index; i < n; i += stride) {
    sca = scale[int(i/exch)];
    max = scalbnf(1,sca + 1) - scalbnf(1,sca+1-man[0]);
    overflow[i] = 0;
    underflow[i] = 0;
    temp = frexpf(data[i], &expo);
    diff = sca - expo + 1;
    if (mn <= diff) {
      temp = scalbnf(temp, mn-1-diff);
      expo = sca - mn + 2;
      diff = mn-1;
    }
    if (0 <= diff) {
      temp = scalbnf(temp, man[diff]);
      if (pre_data[i] > data[i]) temp = ceilf(temp);
      else                       temp = floorf(temp);
      temp = scalbnf(temp, -man[diff]);
    }
    temp = scalbnf(temp, expo);
    if (temp != 0) {
      frexpf(temp, &expo);
      diff = sca - expo + 1;
      if (diff < room) overflow[i] = 1;
      if (diff < room + 1) underflow[i] = 1;
      if (temp > max) temp = max;
      if (temp < -max) temp = -max;
    }
    data[i] = temp;
  }
}

__global__ void fp_quantize_op(int n, float *data, int exp_bit, int man_bit, int bias){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int expo;
  float temp, max;
  max = scalbnf(1,scalbnf(1,exp_bit)-bias) - scalbnf(1,scalbnf(1,exp_bit)-1-bias-man_bit);
  for (int i = index; i < n; i += stride) {
    temp = frexpf(data[i], &expo);
    temp = scalbnf(temp, man_bit+1);
    if (2-bias > expo) {
      temp = scalbnf(temp, expo-2+bias);
      expo = 2-bias;
    }
    temp = scalbnf(rintf(temp), expo-man_bit-1);
    if (temp > max) temp = max;
    if (temp < -max) temp = -max;
    data[i] = temp;
  }
}
__global__ void fp_quantize_sr_op(int n, float *data, int exp_bit, int man_bit, int bias, float *rand){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int expo;
  float temp, max;
  max = scalbnf(1,scalbnf(1,exp_bit)-bias) - scalbnf(1,scalbnf(1,exp_bit)-1-bias-man_bit);
  for (int i = index; i < n; i += stride) {
    temp = frexpf(data[i], &expo);
    temp = scalbnf(temp, man_bit+1);
    if (2-bias > expo) {
      temp = scalbnf(temp, expo-2+bias);
      expo = 2-bias;
    }
    temp = scalbnf(stochastic_round(temp, rand[i]), expo-man_bit-1);
    if (temp > max) temp = max;
    if (temp < -max) temp = -max;
    data[i] = temp;
  }
}
__global__ void fp_hysteresis_op(int n, float *pre_data, float *data, int exp_bit, int man_bit, int bias){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int expo;
  float temp, max;
  max = scalbnf(1,scalbnf(1,exp_bit)-bias) - scalbnf(1,scalbnf(1,exp_bit)-1-bias-man_bit);
  for (int i = index; i < n; i += stride) {
    temp = frexpf(data[i], &expo);
    temp = scalbnf(temp, man_bit+1);
    if (2-bias > expo) {
      temp = scalbnf(temp, expo-2+bias);
      expo = 2-bias;
    }
    if (pre_data[i] > data[i]) temp = ceilf(temp);
    else                       temp = floorf(temp);
    temp = scalbnf(temp, expo-man_bit-1);
    if (temp > max) temp = max;
    if (temp < -max) temp = -max;
    data[i] = temp;
  }
}

__global__ void log4_trim_mantissa_op(int n, float *data, int *scale){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int expo, sca;
  float temp;
  sca = scale[0];
  for (int i = index; i < n; i += stride) {
    temp = frexpf(data[i], &expo);
    if (expo-1 > sca) printf("log4_trim_mantissa function got incorrect scale!");
    else if (expo < sca-5) temp = 0;
    if (temp > 0) temp = scalbnf(0.5, expo);
    else if (temp < 0) temp = scalbnf(-0.5, expo);
    data[i] = temp;
  }
}

void linear_quantize_kernel(int cuda_id, float *data, int bit_num, int *sca, bool *of, bool *uf, int room, int N, int EXCH, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  linear_quantize_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, EXCH, data, bit_num, sca, of, uf, room);
}
void linear_quantize_sr_kernel(int cuda_id, float *data, int bit_num, int *sca, bool *of, bool *uf, int room, float *rand, int N, int EXCH, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  linear_quantize_sr_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, EXCH, data, bit_num, sca, of, uf, room, rand);
}
void linear_hysteresis_kernel(int cuda_id, float *pre_data, float *data, int bit_num, int *sca, bool *of, bool *uf, int room, int N, int EXCH, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  linear_hysteresis_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, EXCH, pre_data, data, bit_num, sca, of, uf, room);
}

void custom_fp_quantize_kernel(int cuda_id, float *data, int *man, int *sca, bool *of, bool *uf, int room, int N, int MN, int EXCH, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  custom_fp_quantize_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, MN, EXCH, data, man, sca, of, uf, room);
}
void custom_fp_quantize_sr_kernel(int cuda_id, float *data, int *man, int *sca, bool *of, bool *uf, int room, float *rand, int N, int MN, int EXCH, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  custom_fp_quantize_sr_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, MN, EXCH, data, man, sca, of, uf, room, rand);
}
void custom_fp_hysteresis_kernel(int cuda_id, float *pre_data, float *data, int *man, int *sca, bool *of, bool *uf, int room, int N, int MN, int EXCH, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  custom_fp_hysteresis_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, MN, EXCH, pre_data, data, man, sca, of, uf, room);
}

void fp_quantize_kernel(int cuda_id, float *data, int exp_bit, int man_bit, int bias, int N, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  fp_quantize_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, data, exp_bit, man_bit, bias);
}
void fp_quantize_sr_kernel(int cuda_id, float *data, int exp_bit, int man_bit, int bias, float *rand, int N, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  fp_quantize_sr_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, data, exp_bit, man_bit, bias, rand);
}
void fp_hysteresis_kernel(int cuda_id, float *pre_data, float *data, int exp_bit, int man_bit, int bias, int N, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  fp_hysteresis_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, pre_data, data, exp_bit, man_bit, bias);
}

void log4_trim_mantissa_kernel(int cuda_id, float *data, int *sca, int N, cudaStream_t stream) {
  cudaSetDevice(cuda_id);
  int BN = GET_BLOCKS(N);
  log4_trim_mantissa_op<<<BN, CUDA_NUM_THREADS, 0, stream>>>(N, data, sca);
}

void linear_quantize_kernel(int cuda_id, float *data, int bit_num, int *sca, bool *of, bool *uf, int room, int N, int EXCH, cudaStream_t stream);
void linear_quantize_sr_kernel(int cuda_id, float *data, int bit_num, int *sca, bool *of, bool *uf, int room, float *rand, int N, int EXCH, cudaStream_t stream);
void linear_hysteresis_kernel(int cuda_id, float *pre_data, float *data, int bit_num, int *sca, bool *of, bool *uf, int room, int N, int EXCH, cudaStream_t stream);

void custom_fp_quantize_kernel(int cuda_id, float *data, int *man, int *sca, bool *of, bool *uf, int room, int N, int MN, int EXCH, cudaStream_t stream);
void custom_fp_quantize_sr_kernel(int cuda_id, float *data, int *man, int *sca, bool *of, bool *uf, int room, float *rand, int N, int MN, int EXCH, cudaStream_t stream);
void custom_fp_hysteresis_kernel(int cuda_id, float *pre_data, float *data, int *man, int *sca, bool *of, bool *uf, int room, int N, int MN, int EXCH, cudaStream_t stream);

void fp_quantize_kernel(int cuda_id, float *data, int exp_bit, int man_bit, int bias, int N, cudaStream_t stream);
void fp_quantize_sr_kernel(int cuda_id, float *data, int exp_bit, int man_bit, int bias, float *rand, int N, cudaStream_t stream);
void fp_hysteresis_kernel(int cuda_id, float *pre_data, float *data, int exp_bit, int man_bit, int bias, int N, cudaStream_t stream);

void log4_trim_mantissa_kernel(int cuda_id, float *data, int *sca, int N, cudaStream_t stream);