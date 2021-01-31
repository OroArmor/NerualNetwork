package com.oroarmor.neural_network.lib.matrix.jcuda.kernels;

import com.oroarmor.neural_network.lib.matrix.jcuda.kernels.functions.AbsKernel;
import com.oroarmor.neural_network.lib.matrix.jcuda.kernels.simpleMath.AddKernel;
import com.oroarmor.neural_network.lib.matrix.jcuda.kernels.simpleMath.AddValueKernel;
import com.oroarmor.neural_network.lib.matrix.jcuda.kernels.simpleMath.MultiplyKernel;
import com.oroarmor.neural_network.lib.matrix.jcuda.kernels.simpleMath.MultiplyValueKernel;
import com.oroarmor.neural_network.lib.util.JCudaKernel;

public abstract class MatrixKernel extends JCudaKernel {

	public MatrixKernel(String name, String subpath) {
		super(name);
		loadKernel("matrixKernels/" + subpath + "/" + name + ".cu");
	}

	public static void rebuildAllKernels() {
		System.out.println("Initializing all kernels...");
		long millis = System.currentTimeMillis();
		AbsKernel.getInstance().rebuildKernel();
		AddKernel.getInstance().rebuildKernel();
		AddValueKernel.getInstance().rebuildKernel();
		MultiplyKernel.getInstance().rebuildKernel();
		MultiplyValueKernel.getInstance().rebuildKernel();
		System.out.println("All kernels initialized in " + (System.currentTimeMillis() - millis) + " milliseconds.");
	}
}
