package com.oroarmor.neural_network.lib.matrix.jcuda.kernels.simpleMath;

import jcuda.Pointer;
import com.oroarmor.neural_network.lib.matrix.jcuda.JCudaMatrix;
import com.oroarmor.neural_network.lib.matrix.jcuda.kernels.MatrixKernel;
import com.oroarmor.neural_network.lib.util.Dim3;

public class AddKernel extends MatrixKernel {

	private static AddKernel instance;

	private AddKernel() {
		super("add", "simpleMath");
	}

	public void add(JCudaMatrix a, JCudaMatrix b, JCudaMatrix out) {
		Dim3 blockSize = new Dim3(1024);
		Dim3 gridSize = new Dim3((int) Math.ceil(a.getCols() * a.getRows() / (double) blockSize.x));

		Pointer params = Pointer.to(a.getSizePointer(), a.getMatrixPointer(), b.getMatrixPointer(),
				out.getMatrixPointer());

		runKernel(params, gridSize, blockSize);
	}

	public static AddKernel getInstance() {
		if (instance == null) {
			instance = new AddKernel();
		}
		return instance;
	}

}
