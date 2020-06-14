extern "C"
__global__ void multiply(double *a, double *b, double *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if( col < k && row < m) 
    {
    	int rowIndex = row * n;
        for(int i = 0; i < n; i++) 
        {
            sum += a[rowIndex + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}