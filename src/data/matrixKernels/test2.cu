extern "C"
#include <stdio.h>

__global__ void test2(int n, double *a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //output x
    
    if(i < n){
	    printf("sizeof i: %d\n", sizeof(i));
	    printf("sizeof a: %d\n", sizeof(a));
	    printf("i = %d\n", i);
    	a[0] = 1;
    }
}
