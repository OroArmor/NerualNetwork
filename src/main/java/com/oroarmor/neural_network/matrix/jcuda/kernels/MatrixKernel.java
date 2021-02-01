package com.oroarmor.neural_network.matrix.jcuda.kernels;

import com.oroarmor.neural_network.matrix.jcuda.kernels.functions.AbsKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath.AddKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath.AddValueKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath.MultiplyKernel;
import com.oroarmor.neural_network.matrix.jcuda.kernels.simpleMath.MultiplyValueKernel;
import com.oroarmor.neural_network.util.JCudaKernel;

/**
 * An abstract class for all matrix kernels
 */
public abstract class MatrixKernel extends JCudaKernel {
    public MatrixKernel(String name, String subPath) {
        super(name);
        loadKernel("matrix_kernels/" + subPath + "/" + name + ".cu");
    }

    /**
     * Rebuilds all kernels
     */
    public static void rebuildAllKernels() {
        System.out.println("Initializing all kernels...");
        long millis = System.currentTimeMillis();
        try {
            AbsKernel.getInstance().rebuildKernel();
            AddKernel.getInstance().rebuildKernel();
            AddValueKernel.getInstance().rebuildKernel();
            MultiplyKernel.getInstance().rebuildKernel();
            MultiplyValueKernel.getInstance().rebuildKernel();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("All kernels initialized in " + (System.currentTimeMillis() - millis) + " milliseconds.");
    }
}
