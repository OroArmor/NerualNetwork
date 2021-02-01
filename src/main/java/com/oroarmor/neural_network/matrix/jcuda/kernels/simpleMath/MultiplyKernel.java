package com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath;

import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.util.Dim3;
import jcuda.Pointer;

/**
 * Multiplies two matrices together
 */
public class MultiplyKernel extends MatrixKernel {
    private static MultiplyKernel instance;

    private MultiplyKernel() {
        super("multiply", "simple_math");
    }

    public static MultiplyKernel getInstance() {
        if (instance == null) {
            instance = new MultiplyKernel();
        }
        return instance;
    }

    /**
     * Multiplies a and b into out
     * @param a Matrix a
     * @param b Matrix b
     * @param out Output
     */
    public void multiply(JCudaMatrix a, JCudaMatrix b, JCudaMatrix out) {
        Dim3 blockSize = new Dim3(16, 64);
        Dim3 gridSize = new Dim3((b.getCols() + blockSize.x - 1) / blockSize.x,
                (a.getRows() + blockSize.y - 1) / blockSize.y);

        Pointer params = Pointer.to(a.getMatrixPointer(), b.getMatrixPointer(), out.getMatrixPointer(),
                a.getRowPointer(), a.getColumnPointer(), b.getColumnPointer());

        runKernel(params, gridSize, blockSize);
    }
}
