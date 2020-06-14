package oroarmor.neuralnetwork.matrix.jcuda.kernels.functions;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.jcuda.JCudaMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.MatrixKernel;
import oroarmor.neuralnetwork.util.Dim3;

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
