"""
This code is to learn how to transfer data from CPU (RAM) 
to GPU (VRAM) using cuda api
"""


# imports for the app
# numpy for data structures like arrays,matrix and tensors....
import numpy as np

# cuda for GPU programming
import pycuda.autoinit
from pycuda import gpuarray

# data on CPU
host_data = np.array([1,2,3,4,5],dtype=np.float32)

# copy data to GPU
device_data = gpuarray.to_gpu(host_data)

# do some arthimetic to change data 
device_data_x2 = 2 * device_data
host_data_x2 = device_data_x2.get()

# print to show data
print(host_data_x2)
