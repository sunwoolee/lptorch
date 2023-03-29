#include <algorithm>

constexpr int CUDA_NUM_THREADS = 1024;

constexpr int MAXIMUM_NUM_BLOCKS = 2048;

constexpr int MAXIMUM_REDUCTION_NUM_BLOCKS = 256;

inline int GET_BLOCKS(const int N) {
    return std::max(std::min((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, MAXIMUM_NUM_BLOCKS), 1); }

inline int GET_REDUCTION_BLOCKS(const int N) {
    return std::max(std::min((N + 2*CUDA_NUM_THREADS - 1) / (2*CUDA_NUM_THREADS), MAXIMUM_REDUCTION_NUM_BLOCKS), 1); }

