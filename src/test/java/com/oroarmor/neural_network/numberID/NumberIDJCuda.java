/*
 * MIT License
 *
 * Copyright (c) 2021 OroArmor
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.oroarmor.neural_network.numberID;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.util.Dim3;
import com.oroarmor.neural_network.util.JCudaHelper;
import com.oroarmor.neural_network.util.JCudaKernel;
import jcuda.Pointer;

public class NumberIDJCuda {
    private static JCudaKernel testKernel;
    private static JCudaMatrix matrix;
    private static JCudaMatrix matrix2;
    private static CPUMatrix cpuMatrix;
    private static CPUMatrix cpuMatrix2;

    public static void main(String[] args) {
        JCudaHelper.InitJCuda(true);
        MatrixKernel.rebuildAllKernels();
        testKernel = new JCudaKernel("test2");
        testKernel.loadKernel("matrixKernels/test2.cu");

        JCudaMatrix.randomMatrix(10, 10, null, 0, 10);

        initializeMatrices();
        for (int i = 1; i < 100000; i *= 10)
            testImprovement(i);
    }

    private static void testImprovement(int ops) {
        long millis = System.currentTimeMillis();
        for (int i = 0; i < ops; i++) {
            cpuMatrix.multiplyMatrix(cpuMatrix2);
        }
        long cpuTime = (System.currentTimeMillis() - millis);
        System.out.println("CPU: " + cpuTime);

        millis = System.currentTimeMillis();
        for (int i = 0; i < ops; i++) {
            matrix.multiplyMatrix(matrix2);
        }
        long gpuTime = (System.currentTimeMillis() - millis);
        System.out.println("GPU: " + gpuTime);

        System.out.printf("Improvement at %d ops: %.2f%%\n", ops, (double) 100 * cpuTime / gpuTime);
    }

    private static void initializeMatrices() {
        int dims = 1 << 6;

        matrix = new JCudaMatrix(dims, dims).keep();
        matrix2 = new JCudaMatrix(dims, dims).keep();

        Dim3 blockSize = new Dim3(1024);
        Dim3 gridSize = new Dim3((int) Math.ceil(matrix.getCols() * matrix.getRows() / (double) blockSize.x));

        Pointer params = Pointer.to(matrix.getSizePointer(), matrix.getMatrixPointer());
        testKernel.runKernel(params, gridSize, blockSize);

        params = Pointer.to(matrix2.getSizePointer(), matrix2.getMatrixPointer());
        testKernel.runKernel(params, gridSize, blockSize);

        cpuMatrix = matrix.toCPUMatrix();
        cpuMatrix2 = matrix2.toCPUMatrix();
    }
}
