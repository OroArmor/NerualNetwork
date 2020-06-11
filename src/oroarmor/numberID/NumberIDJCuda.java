package oroarmor.numberID;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.JCudaMatrix;
import oroarmor.neuralnetwork.matrix.MatrixKernel;
import oroarmor.neuralnetwork.util.Dim3;

public class NumberIDJCuda {

	public static void main(String[] args) {

		MatrixKernel.InitJCuda(true);

		MatrixKernel test = new MatrixKernel("test2");

		test.loadKernel("src/data/matrixKernels/test2.cu");

		JCudaMatrix testM = new JCudaMatrix(1, 1);

		testM.createPointers();

		Dim3 blockSize = new Dim3(16);
		Dim3 gridSize = new Dim3((int) Math.ceil(testM.getCols() * testM.getRows() / (double) blockSize.x));

		System.out.println(blockSize);
		System.out.println(gridSize);

		Pointer params = Pointer.to(testM.getSizePointer(), testM.getMatrixPointer());

		System.out.println(params);

		test.runKernel(params, gridSize, blockSize);
	}

}
