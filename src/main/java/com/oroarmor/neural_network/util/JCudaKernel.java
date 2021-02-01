package com.oroarmor.neural_network.util;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.*;

/**
 * A JCuda kernel
 * @author OroArmor
 */
public class JCudaKernel {
    /**
     * The cuda function for this kernel
     */
    protected final CUfunction function = new CUfunction();

    /**
     * The name of the kernel
     */
    protected final String name;

    /**
     * The path to the cu file
     */
    protected String pathToCuFile;

    /**
     * Creates a new {@link JCudaKernel}
     * @param name The name of the kernel
     */
    public JCudaKernel(String name) {
        this.name = name;
    }

    /**
     * Loads the cu file and creates the kernel
     * @param kernelPath The path to the kernel file
     */
    public void loadKernel(String kernelPath) {
        checkInit();
        pathToCuFile = kernelPath;
        String cubinFileName = JCudaHelper.prepareDefaultCubinFile(kernelPath);
        cuModuleLoad(JCudaHelper.module, cubinFileName);
        cuModuleGetFunction(function, JCudaHelper.module, name);
    }

    /**
     * Runs the cuda kernel
     * @param parameters The pointer to the parameters of the kernel
     * @param gridSize The grid size for this run
     * @param blockSize The block size for this run
     */
    public void runKernel(Pointer parameters, Dim3 gridSize, Dim3 blockSize) {
        cuLaunchKernel(//
                function, // Kernel Function
                gridSize.x, gridSize.y, gridSize.z, // Grid dimension
                blockSize.x, blockSize.y, blockSize.z, // Block dimension
                0, null, // Shared memory size and stream
                parameters, null // Kernel- and extra parameters
        );

        cuCtxSynchronize();
    }

    /**
     * Checks if JCuda was initialized
     */
    protected void checkInit() {
        if (!JCudaHelper.INIT) {
            System.out.println("JCuda was initialized by the program.");
            JCudaHelper.InitJCuda(true);
        }
    }

    /**
     * Rebuilds the kernel
     */
    protected void rebuildKernel() {
        checkInit();
        String cubinFileName = JCudaHelper.prepareDefaultCubinFile(pathToCuFile);
        cuModuleLoad(JCudaHelper.module, cubinFileName);
        cuModuleGetFunction(function, JCudaHelper.module, name);
    }
}
