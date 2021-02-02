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

package com.oroarmor.neural_network.matrix.jcuda.kernels.functions;

import java.util.Random;

import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;

/**
 * A matrix kernel that puts random values into a matrix
 * @author OroArmor
 */
public class RandomKernel extends MatrixKernel {
    private static RandomKernel instance;
    private static curandGenerator generator;

    private RandomKernel() {
        super("random_matrix", "functions");
    }

    public static RandomKernel getInstance() {
        JCurand.setExceptionsEnabled(true);

        generator = new curandGenerator();
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator, 1234);

        if (instance == null) {
            instance = new RandomKernel();
        }
        return instance;
    }

    /**
     * Puts random values from min to max in out
     *
     * @param out    The output matrix
     * @param random The random number generator to set the seed
     * @param min    The minimum value
     * @param max    The maximum value
     */
    public void random(JCudaMatrix out, Random random, double min, double max) {
        if (random != null) {
            curandSetPseudoRandomGeneratorSeed(generator, random.nextLong());
        }
        curandGenerateUniformDouble(generator, out.getDeviceMPtr(), out.getCols() * out.getRows());
        out.multiply(max - min).add(min);
    }
}
