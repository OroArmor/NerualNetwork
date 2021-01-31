package com.oroarmor.neural_network.lib.matrix.jcuda.kernels.functions;

import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;

import java.util.Random;

import jcuda.Pointer;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import com.oroarmor.neural_network.lib.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.lib.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.lib.util.Dim3;

public class RandomKernel extends MatrixKernel {

	private static RandomKernel instance;
	private static curandGenerator generator;

	private RandomKernel() {
		super("random_matrix", "functions");
	}

	public void random(JCudaMatrix out, Random random, double min, double max) {
		if (random != null) {
			curandSetPseudoRandomGeneratorSeed(generator, random.nextLong());
		}
		curandGenerateUniformDouble(generator, out.getDeviceMPtr(), out.getCols() * out.getRows());
		out.multiply(max - min).add(min);
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

	@Override
	public void loadKernel(String kernelPath) {
	}

	@Override
	public void runKernel(Pointer parameters, Dim3 gridSize, Dim3 blockSize) {
	}

	@Override
	public void checkInit() {
	}

	@Override
	public void rebuildKernel() {
	}

}
