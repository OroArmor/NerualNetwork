package com.oroarmor.neural_network.lib.matrix.jcuda.kernels.functions;

import jcuda.Pointer;
import com.oroarmor.neural_network.lib.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.lib.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.lib.util.Dim3;

public class AbsKernel extends MatrixKernel {

	private static AbsKernel instance;

	private AbsKernel() {
		super("abs_value", "functions");
	}

	public void abs(JCudaMatrix a, JCudaMatrix out) {
		Dim3 blockSize = new Dim3(1024);
		Dim3 gridSize = new Dim3((int) Math.ceil(a.getCols() * a.getRows() / (double) blockSize.x));

		Pointer params = Pointer.to(a.getSizePointer(), a.getMatrixPointer(), out.getMatrixPointer());

		runKernel(params, gridSize, blockSize);
	}

	public static AbsKernel getInstance() {
		if (instance == null) {
			instance = new AbsKernel();
		}
		return instance;
	}

}
