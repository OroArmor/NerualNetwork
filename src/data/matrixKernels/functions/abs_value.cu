extern "C"
#include <stdlib.h>
__global__ void abs_value(int n, double *a, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //output x
    if (i < n)
    {
    	c[i] = abs(a[i]);
    }
}