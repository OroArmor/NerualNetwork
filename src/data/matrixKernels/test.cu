extern "C"
__global__ void test(int n, double *a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //output x
    if (i < n)
    {
    	a[i] = 1;
    }
}
