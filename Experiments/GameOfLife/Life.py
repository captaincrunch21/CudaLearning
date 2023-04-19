import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from time import time
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# C Code for GPU
Life_Kernal_module = SourceModule("""

#define _X ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y ( threadIdx.y + blockIdx.y * blockDim.y )

#define _WIDTH 4096
#define _HEIGHT 4096

#define _XM(x) ( (x + _WIDTH) % _WIDTH )
#define _YM(y) ( (y + _HEIGHT) % _HEIGHT )



#define _INDEX(x,y) ( _XM(x) + _YM(y) * _WIDTH  )

__device__ int nbrs(int x, int y, int * in)
{
 return ( in[ _INDEX(x -1, y+1) ] + in[ _INDEX(x-1, y) ] + in[ _INDEX(x-1, y-1) ] \
 + in[ _INDEX(x, y+1)] + in[_INDEX(x, y - 1)] \
 + in[ _INDEX(x+1, y+1) ] + in[ _INDEX(x+1, y) ] + in[
_INDEX(x+1, y-1) ] );
}

__global__ void conway_ker(int * lattice_out, int * lattice )
{
 /* x, y are the appropriate values for the cell covered by this thread*/
 int x = _X, y = _Y;
 
 /* count the number of neighbors around the current cell*/
 int n = nbrs(x, y, lattice);
 
 
 /* if the current cell is alive, then determine if it lives or dies for the next generation.*/
 if ( lattice[_INDEX(x,y)] == 1)
 switch(n)
 {
 /* if the cell is alive: it remains alive only if it has 2 or 3
neighbors.*/
 case 2:
 case 3: lattice_out[_INDEX(x,y)] = 1;
 break;
 default: lattice_out[_INDEX(x,y)] = 0; 
 }
 else if( lattice[_INDEX(x,y)] == 0 )
 switch(n)
 {
 /* a dead cell comes to life only if it has 3 neighbors that are
alive.*/
 case 3: lattice_out[_INDEX(x,y)] = 1;
 break;
 default: lattice_out[_INDEX(x,y)] = 0; 
 }
}
""")

# Image description
width, height = 4096, 4096
# Get function from Kernal
conway_ker = Life_Kernal_module.get_function("conway_ker")

def update_gpu(frameNum, img, newLattice_gpu,lattice_gpu,N):
    # only 1024 threads are available for block so used 32x32
    conway_ker(newLattice_gpu,lattice_gpu,grid=(N//32,N//32,1),block=(32,32,1))
    # Get image Data from GPU
    img.set_data(newLattice_gpu.get())
    # swap buffers
    lattice_gpu.set(newLattice_gpu)
    return img
    
if __name__ == '__main__':
    N = 4096
    lattice = np.int32(np.random.choice([1,0],N*N , p=[0.25,0.75]).reshape(N,N))
    lattice_gpu = gpuarray.to_gpu(lattice)
    newLattice_gpu = gpuarray.empty_like(lattice_gpu)
    fig,ax = plt.subplots()
    img = ax.imshow(lattice_gpu.get(), interpolation='nearest')
    # Animation of all states in Life
    ani = FuncAnimation(fig,update_gpu,fargs=(img,newLattice_gpu,lattice_gpu,N),interval=0,frames =1000)
    plt.show()