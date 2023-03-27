Welcome to the lptorch wiki!

lptorch is CUDA extension library of pytorch for low precision neural network training / inference.

We developed lptorch to verify NPU (real hardware), so it use only 2^N scale factor which can be implemented with bit shift operation in hardware.

# Install_requires
* torch (with cuda)
* onnx

# Installation
```
git clone https://github.com/sunwoolee/lptorch.git
cd lptorch
make clean
make run
```

# Installation with multiple CUDA architectures
Check your GPU compute capability in https://developer.nvidia.com/cuda-gpus#compute  
```
export TORCH_CUDA_ARCH_LIST="{your GPU compute capability}"
# export TORCH_CUDA_ARCH_LIST="6.1 7.5 8.6"
```

See the wiki for more information on the package.
