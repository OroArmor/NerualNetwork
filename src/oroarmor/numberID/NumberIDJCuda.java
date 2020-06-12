package oroarmor.numberID;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.JCudaMatrix;
import oroarmor.neuralnetwork.matrix.MatrixKernel;
import oroarmor.neuralnetwork.util.Dim3;

public class NumberIDJCuda {

	public static void main(String[] args) {
		MatrixKernel.InitJCuda(true);

		MatrixKernel testKernel = new MatrixKernel("test2");

		testKernel.loadKernel("src/data/matrixKernels/test2.cu");

		JCudaMatrix matrix = new JCudaMatrix(16, 1);

		Dim3 blockSize = new Dim3(16);
		Dim3 gridSize = new Dim3((int) Math.ceil(matrix.getCols() * matrix.getRows() / (double) blockSize.x));

		Pointer params = Pointer.to(matrix.getSizePointer(), matrix.getMatrixPointer());

		testKernel.runKernel(params, gridSize, blockSize);

		matrix.toCPUMatrix().print();
	}

}
