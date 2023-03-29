#include <ATen/cuda/CUDAContext.h>
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