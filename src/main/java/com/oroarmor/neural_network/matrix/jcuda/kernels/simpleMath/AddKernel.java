package com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath;

import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.util.Dim3;
import jcuda.Pointer;

/**
 * Adds two matrices together
 * @author OroArmor
 */
public class AddKernel extends MatrixKernel {
    private static AddKernel instance;

    private AddKernel() {
        super("add", "simple_math");
    }

    public static AddKernel getInstance() {
        if (instance == null) {
            instance = new AddKernel();
        }
        return instance;
    }

    /**
     * Adds a and b into out
     * @param a Matrix a
     * @param b Matrix b
     * @param out Output
     */
    public void add(JCudaMatrix a, JCudaMatrix b, JCudaMatrix out) {
        Dim3 blockSize = new Dim3(1024);
        Dim3 gridSize = new Dim3((int) Math.ceil(a.getCols() * a.getRows() / (double) blockSize.x));

        Pointer params = Pointer.to(a.getSizePointer(), a.getMatrixPointer(), b.getMatrixPointer(),
                out.getMatrixPointer());

        runKernel(params, gridSize, blockSize);
    }
}
