package oroarmor.numberID;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.JCudaMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.MatrixKernel;
import oroarmor.neuralnetwork.util.Dim3;
import oroarmor.neuralnetwork.util.JCudaKernel;

public class NumberIDJCuda {

	private static JCudaKernel testKernel;
	private static JCudaMatrix matrix;
	private static JCudaMatrix matrix2;
	private static CPUMatrix cpuMatrix;
	private static CPUMatrix cpuMatrix2;

	public static void main(String[] args) {
		JCudaKernel.InitJCuda(true);
		testKernel = new JCudaKernel("test2");
		testKernel.loadKernel("src/data/matrixKernels/test2.cu");
		initializeMatricies();
		for (int i = 0; i < 10; i++)
			testImprovement();
	}

	private static void testImprovement() {
		int ops = 10000;

		long millis = System.currentTimeMillis();
		for (int i = 0; i < ops; i++) {
			cpuMatrix = cpuMatrix.multiplyMatrix(cpuMatrix2);
		}
		long cpuTime = (System.currentTimeMillis() - millis);
		System.out.println("CPU: " + cpuTime);

		millis = System.currentTimeMillis();
		for (int i = 0; i < ops; i++) {
			matrix = matrix.multiplyMatrix(matrix2);
		}
		long gpuTime = (System.currentTimeMillis() - millis);
		System.out.println("GPU: " + gpuTime);

		System.out.printf("Improvement: %.2f%%\n", (double) 100 * cpuTime / gpuTime);
	}

	private static void initializeMatricies() {
		int dims = 1 << 2;

		matrix = new JCudaMatrix(dims, dims).keep();
		matrix2 = new JCudaMatrix(dims, dims).keep();

		Dim3 blockSize = new Dim3(1024);
		Dim3 gridSize = new Dim3((int) Math.ceil(matrix.getCols() * matrix.getRows() / (double) blockSize.x));

		Pointer params = Pointer.to(matrix.getSizePointer(), matrix.getMatrixPointer());
		testKernel.runKernel(params, gridSize, blockSize);

		params = Pointer.to(matrix2.getSizePointer(), matrix2.getMatrixPointer());
		testKernel.runKernel(params, gridSize, blockSize);

		cpuMatrix = matrix.toCPUMatrix();
		cpuMatrix2 = matrix2.toCPUMatrix();

		MatrixKernel.initializeAllKernels();

		matrix.abs().toCPUMatrix().print();
		matrix2.toCPUMatrix().print();
	}

}
