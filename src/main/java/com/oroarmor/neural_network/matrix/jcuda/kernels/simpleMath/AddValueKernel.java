package com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath;

import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.util.Dim3;
import jcuda.Pointer;

/**
 * Adds a value to all values in a matrix
 * @author OroArmor
 */
public class AddValueKernel extends MatrixKernel {
    private static AddValueKernel instance;

    private AddValueKernel() {
        super("add_value", "simple_math");
    }

    public static AddValueKernel getInstance() {
        if (instance == null) {
            instance = new AddValueKernel();
        }
        return instance;
    }

    /**
     * Adds b to all values in a
     * @param a The matrix
     * @param b The value
     * @param out The output
     */
    public void add(JCudaMatrix a, double b, JCudaMatrix out) {
        Dim3 blockSize = new Dim3(1024);
        Dim3 gridSize = new Dim3((int) Math.ceil(a.getCols() * a.getRows() / (double) blockSize.x));

        Pointer params = Pointer.to(a.getSizePointer(), a.getMatrixPointer(), Pointer.to(new double[]{b}),
                out.getMatrixPointer());

        runKernel(params, gridSize, blockSize);
    }
}
