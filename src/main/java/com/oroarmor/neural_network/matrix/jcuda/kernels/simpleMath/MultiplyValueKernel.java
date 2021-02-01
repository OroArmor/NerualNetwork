package com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath;

import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.util.Dim3;
import jcuda.Pointer;

/**
 * Multiplies all values in a matrix
 * @author OroArmor
 */
public class MultiplyValueKernel extends MatrixKernel {
    private static MultiplyValueKernel instance;

    private MultiplyValueKernel() {
        super("multiply_value", "simple_math");
    }

    public static MultiplyValueKernel getInstance() {
        if (instance == null) {
            instance = new MultiplyValueKernel();
        }
        return instance;
    }

    /**
     * Multiply all values in a by b
     * @param a Matrix a
     * @param b The value
     * @param out The output
     */
    public void multiplyValue(JCudaMatrix a, double b, JCudaMatrix out) {
        Dim3 blockSize = new Dim3(1024);
        Dim3 gridSize = new Dim3((int) Math.ceil(a.getCols() * a.getRows() / (double) blockSize.x));

        Pointer params = Pointer.to(a.getSizePointer(), a.getMatrixPointer(), Pointer.to(new double[]{b}),
                out.getMatrixPointer());

        runKernel(params, gridSize, blockSize);
    }
}
