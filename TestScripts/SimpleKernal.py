"""
This class is to write a simple kernal 
for GPU
"""
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from time import time
from pycuda.compiler import SourceModule


# C Code to run on GPU
# computing thread id by using block data 
# using only X ---> one dimensional
kernal_code = SourceModule("""
__global__ void double_kernal(float* in,float* out)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        out[tid] = 2 * in[tid];
    }
""")

# getting function symbols
kernal = kernal_code.get_function("double_kernal")

# some function to compute speed
def speedcomparison():
    # data initialize for CPU
    n = 500000000
    host_data = np.float32(np.random.random(n))
    t1 = time()
    host_data_2x = host_data * np.float32(2)
    t2 = time()
    print('total time to compute on CPU:', t2-t1)
    # copying from CPU to GPU
    device_data = gpuarray.to_gpu(host_data)
    # allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data)
    # number of threads in each block
    block_size = 256
    # number of blocks required
    grid_size = (n + block_size -1) // block_size 
    t1 = time()
    # Run PyCUDA kernel code here
    # Using grids and blocks to be one dimensional
    kernal(device_data,device_data_2x, block=(block_size,1,1),grid=(grid_size,1,1))
    t2 = time()
    from_device = device_data_2x.get()
    print('total time to compute on GPU:', t2-t1)
    



# Calling Costly function
speedcomparison()
