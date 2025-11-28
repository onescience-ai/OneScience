import os
from setuptools import setup, Extension, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, load

# os.environ["USE_ROCM"] = "1"

        
def compile(name, sources, extra_include_paths, build_directory):
    # cmd = 'source ${ROCM_PATH}/cuda/env.sh'
    # result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            '-std=c++17',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
        ],
        verbose=True,
        build_directory=build_directory,
    )
