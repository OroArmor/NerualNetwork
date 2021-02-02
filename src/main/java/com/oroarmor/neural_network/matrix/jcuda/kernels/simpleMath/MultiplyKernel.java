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
