package com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath;

import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.util.Dim3;
import jcuda.Pointer;

public class MultiplyValueKernel extends MatrixKernel {
    private static MultiplyValueKernel instance;

    private MultiplyValueKernel() {
        super("multiplyValue", "simpleMath");
    }

    public static MultiplyValueKernel getInstance() {
        if (instance == null) {
            instance = new MultiplyValueKernel();
        }
        return instance;
    }

    public void multiplyValue(JCudaMatrix a, double b, JCudaMatrix out) {
        Dim3 blockSize = new Dim3(1024);
        Dim3 gridSize = new Dim3((int) Math.ceil(a.getCols() * a.getRows() / (double) blockSize.x));

        Pointer params = Pointer.to(a.getSizePointer(), a.getMatrixPointer(), Pointer.to(new double[]{b}),
                out.getMatrixPointer());

        runKernel(params, gridSize, blockSize);
    }
}
