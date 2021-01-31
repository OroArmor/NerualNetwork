extern "C"
__global__ void multiplyValue(int n, double *a, double b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //output x
    if (i < n)
    {
    	c[i] = a[i] * b;
    }
}