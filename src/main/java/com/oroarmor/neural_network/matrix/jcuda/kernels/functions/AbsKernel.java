package com.oroarmor.neural_network.matrix.jcuda.kernels.functions;

import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.util.Dim3;
import jcuda.Pointer;

/**
 * A matrix kernel that runs the abs on all values
 * @author OroArmor
 */
public class AbsKernel extends MatrixKernel {
    private static AbsKernel instance;

    private AbsKernel() {
        super("abs_value", "functions");
    }

    public static AbsKernel getInstance() {
        if (instance == null) {
            instance = new AbsKernel();
        }
        return instance;
    }

    /**
     * Gets the absolute values from a and puts them in out
     * @param a The input matrix
     * @param out The output matrix
     */
    public void abs(JCudaMatrix a, JCudaMatrix out) {
        Dim3 blockSize = new Dim3(1024);
        Dim3 gridSize = new Dim3((int) Math.ceil(a.getCols() * a.getRows() / (double) blockSize.x));

        Pointer params = Pointer.to(a.getSizePointer(), a.getMatrixPointer(), out.getMatrixPointer());

        runKernel(params, gridSize, blockSize);
    }
}
