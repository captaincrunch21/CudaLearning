"""
This program does compare speed difference between
CPU and GPU while performing the same operation
"""
# numpy for the arrays
import numpy as np
# initializes the PyCUDA runtime and provides access to the CUDA driver API
import pycuda.autoinit
# provides an interface for creating and manipulating arrays on the GPU
from pycuda import gpuarray
# to Count time
from time import time

# data initialize for CPU
host_data = np.float32(np.random.random(50000000))
# start time CPU
t1 = time()
# some costly operation
host_data_2x = host_data * np.float32(2)
# end time CPU
t2 = time()
print('total time to compute on CPU:', t2-t1)
# copy data to GPU
device_data = gpuarray.to_gpu(host_data)
# GPU start time
t1 =time()
# some costly operation
device_data_2x = device_data * np.float32(2)
# GPU end time
t2 = time()
# copy data back to CPU
from_device = device_data_2x.get()
print('total time to compute on GPU:', t2-t1)





