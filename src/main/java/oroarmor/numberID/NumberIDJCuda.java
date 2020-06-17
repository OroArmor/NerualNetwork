package oroarmor.numberID;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.JCudaMatrix;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.functions.RandomKernel;
import oroarmor.neuralnetwork.util.Dim3;
import oroarmor.neuralnetwork.util.JCudaHelper;
import oroarmor.neuralnetwork.util.JCudaKernel;

public class NumberIDJCuda {

	private static JCudaKernel testKernel;
	private static JCudaMatrix matrix;
	private static JCudaMatrix matrix2;
	private static CPUMatrix cpuMatrix;
	private static CPUMatrix cpuMatrix2;

	public static void main(String[] args) {
		JCudaHelper.InitJCuda(true);
		testKernel = new JCudaKernel("test2");
		testKernel.loadKernel("src/data/matrixKernels/test2.cu");

		RandomKernel.getInstance();

		initializeMatricies();
		for (int i = 1; i < 100000; i *= 10)
			testImprovement(i);
	}

	private static void testImprovement(int ops) {

		long millis = System.currentTimeMillis();
		for (int i = 0; i < ops; i++) {
			cpuMatrix.multiplyMatrix(cpuMatrix2);
		}
		long cpuTime = (System.currentTimeMillis() - millis);
		System.out.println("CPU: " + cpuTime);

		millis = System.currentTimeMillis();
		for (int i = 0; i < ops; i++) {
			matrix.multiplyMatrix(matrix2);
		}
		long gpuTime = (System.currentTimeMillis() - millis);
		System.out.println("GPU: " + gpuTime);

		System.out.printf("Improvement at %d ops: %.2f%%\n", ops, (double) 100 * cpuTime / gpuTime);
	}

	private static void initializeMatricies() {
		int dims = 1 << 6;

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
	}
}
