package com.oroarmor.neural_network.lib.util;

import static jcuda.driver.JCudaDriver.*;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class JCudaKernel {

	private CUfunction function = new CUfunction();
	private String name;
	private String pathToCuFile;

	public JCudaKernel(String name) {
		this.name = name;
	}

	public void loadKernel(String kernelPath) {
		checkInit();
		pathToCuFile = kernelPath;
		String cubinFileName = JCudaHelper.prepareDefaultCubinFile(kernelPath);
		cuModuleLoad(JCudaHelper.module, cubinFileName);
		cuModuleGetFunction(function, JCudaHelper.module, name);
	}

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

	public void checkInit() {
		if (!JCudaHelper.INIT) {
			System.out.println("JCuda was initialized by the program.");
			JCudaHelper.InitJCuda(true);
		}
	}

	public void rebuildKernel() {
		checkInit();
		String cubinFileName = JCudaHelper.prepareDefaultCubinFile(pathToCuFile);
		cuModuleLoad(JCudaHelper.module, cubinFileName);
		cuModuleGetFunction(function, JCudaHelper.module, name);
	}
}
