"""
This class is to write a simple kernal 
for GPU
"""
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel
gpu_2x_ker = ElementwiseKernel(
" float *in, float *out",
"out[i] = 2*in[i];",
"gpu_2x_ker") 

# some function to compute speed
def speedcomparison():
    # data initialize for CPU
    host_data = np.float32(np.random.random(5000000000))
    t1 = time()
    host_data_2x = host_data * np.float32(2)
    t2 = time()
    print('total time to compute on CPU:', t2-t1)
    device_data = gpuarray.to_gpu(host_data)
    # allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data)
    t1 = time()
    gpu_2x_ker(device_data,device_data_2x)
    t2 = time()
    from_device = device_data_2x.get()
    print('total time to compute on GPU:', t2-t1)
 
if __name__ == '__main__':
    speedcomparison()
    