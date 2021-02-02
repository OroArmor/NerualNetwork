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
