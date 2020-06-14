package oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.jcuda.JCudaMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.MatrixKernel;
import oroarmor.neuralnetwork.util.Dim3;

public class MultiplyKernel extends MatrixKernel {

	private static MultiplyKernel instance;

	private MultiplyKernel() {
		super("multiply", "simpleMath");
	}

	public void multiply(JCudaMatrix a, JCudaMatrix b, JCudaMatrix out) {
		Dim3 blockSize = new Dim3(16, 64);
		Dim3 gridSize = new Dim3((b.getCols() + blockSize.x - 1) / blockSize.x,
				(a.getRows() + blockSize.y - 1) / blockSize.y);

		Pointer params = Pointer.to(a.getMatrixPointer(), b.getMatrixPointer(), out.getMatrixPointer(),
				a.getRowPointer(), a.getColumnPointer(), b.getColumnPointer());

		runKernel(params, gridSize, blockSize);
	}

	public static MultiplyKernel getInstance() {
		if (instance == null) {
			instance = new MultiplyKernel();
		}
		return instance;
	}

}
