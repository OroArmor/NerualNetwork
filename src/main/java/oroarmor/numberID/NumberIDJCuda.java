package oroarmor.numberID;

import jcuda.Pointer;
import jcuda.driver.JCudaDriver;
import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.JCudaMatrix;
import oroarmor.neuralnetwork.util.Dim3;
import oroarmor.neuralnetwork.util.JCudaKernel;

public class NumberIDJCuda {

	public static void main(String[] args) {
		JCudaKernel.InitJCuda(true);

		long[] free = new long[1], total = new long[1];
		JCudaDriver.cuMemGetInfo(free, total);
		System.out.println("Max matricies: " + free[0] / (1024 * 1024 * 8));

		JCudaKernel testKernel = new JCudaKernel("test2");

		testKernel.loadKernel("src/data/matrixKernels/test2.cu");

		int dims = 1 << 2;

		JCudaMatrix matrix = new JCudaMatrix(dims, dims).keep();
		JCudaMatrix matrix2 = new JCudaMatrix(dims, dims).keep();

		Dim3 blockSize = new Dim3(1024);
		Dim3 gridSize = new Dim3((int) Math.ceil(matrix.getCols() * matrix.getRows() / (double) blockSize.x));

		Pointer params = Pointer.to(matrix.getSizePointer(), matrix.getMatrixPointer());
		testKernel.runKernel(params, gridSize, blockSize);

		params = Pointer.to(matrix2.getSizePointer(), matrix2.getMatrixPointer());
		testKernel.runKernel(params, gridSize, blockSize);

		CPUMatrix cpuMatrix = matrix.toCPUMatrix();
		CPUMatrix cpuMatrix2 = matrix2.toCPUMatrix();

//		matrix.toCPUMatrix().print();

		int adds = 100;

		long millis = System.currentTimeMillis();
		for (int i = 0; i < adds; i++) {
			cpuMatrix.addMatrix(cpuMatrix2);
		}
		long cpuTime = (System.currentTimeMillis() - millis);
		System.out.println("CPU: " + cpuTime);

		millis = System.currentTimeMillis();
		for (int i = 0; i < adds; i++) {
			matrix.addMatrix(matrix2);
		}

		matrix.add(1).toCPUMatrix().print();

		long gpuTime = (System.currentTimeMillis() - millis);
		System.out.println("GPU: " + gpuTime);
		System.out.printf("Improved by: %.2f%%", (double) 100 * cpuTime / gpuTime);
	}

}
