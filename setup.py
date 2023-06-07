from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='lptorch',
    install_requires=['torch', 'onnx'],
    packages=['lptorch'],
    #package_dir={'fp8': 'C:\\Users\\User\\8bit_training\\cuda\\'},
    package_dir={'lptorch': './'},
    py_modules=['__init__','major','modules','optim', 'quant', 'functions', 'model_wrapper', 'analysis'],
    ext_modules=[
        CUDAExtension('lptorch_cuda', [
            'bind.cpp',
            'lptorch.cu',
        ])
    ],
    cmdclass={
        'build_ext':BuildExtension
    }
)
