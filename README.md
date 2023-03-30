# lptorch
lptorch is CUDA extension library of pytorch for low precision neural network training / inference.

lptorch only supports $2^N$ scale factor which can be implemented with bit shift operation in hardware.

# Install_requires
* torch (with cuda)
* onnx

# Installation
```bash
git clone https://github.com/sunwoolee/lptorch.git
cd lptorch
make clean
make run
```

# Installation with multiple CUDA architectures
Check your GPU compute capability at https://developer.nvidia.com/cuda-gpus#compute  
```bash
export TORCH_CUDA_ARCH_LIST="{your GPU compute capability}"
# export TORCH_CUDA_ARCH_LIST="6.1 7.5 8.6"
```
Follow **Installation** in the latest cuda architecture environment.  

See the [wiki](https://github.com/sunwoolee/lptorch/wiki) for more information about the package.
