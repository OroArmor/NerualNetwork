package com.oroarmor.neural_network.matrix.jcuda.kernels.functions;

import java.util.Random;

import com.oroarmor.neural_network.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.matrix.jcuda.kernels.MatrixKernel;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;

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

    public void random(JCudaMatrix out, Random random, double min, double max) {
        if (random != null) {
            curandSetPseudoRandomGeneratorSeed(generator, random.nextLong());
        }
        curandGenerateUniformDouble(generator, out.getDeviceMPtr(), out.getCols() * out.getRows());
        out.multiply(max - min).add(min);
    }
}
