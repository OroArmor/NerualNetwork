package oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.jcuda.JCudaMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.MatrixKernel;
import oroarmor.neuralnetwork.util.Dim3;

public class AddValueKernel extends MatrixKernel {

	private static AddValueKernel instance;

	private AddValueKernel() {
		super("addValue", "simpleMath");
	}

	public void add(JCudaMatrix a, double b, JCudaMatrix out) {
		Dim3 blockSize = new Dim3(1024);
		Dim3 gridSize = new Dim3((int) Math.ceil(a.getCols() * a.getRows() / (double) blockSize.x));

		Pointer params = Pointer.to(a.getSizePointer(), a.getMatrixPointer(), Pointer.to(new double[] { b }),
				out.getMatrixPointer());

		runKernel(params, gridSize, blockSize);
	}

	public static AddValueKernel getInstance() {
		if (instance == null) {
			instance = new AddValueKernel();
		}
		return instance;
	}

}
