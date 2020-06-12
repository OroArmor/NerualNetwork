package oroarmor.neuralnetwork.matrix.jcuda.kernels;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.jcuda.JCudaMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.MatrixKernel;
import oroarmor.neuralnetwork.util.Dim3;

public class AddKernel extends MatrixKernel {

	private static AddKernel instance;

	private AddKernel() {
		super("add");
		this.loadKernel("src/data/matrixKernels/add.cu");
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
