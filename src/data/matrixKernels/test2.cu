extern "C"
#include <stdio.h>

__global__ void test2(int n, double *a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //output x
    
    if(i < n){
    	a[i] = i;
    }
}
