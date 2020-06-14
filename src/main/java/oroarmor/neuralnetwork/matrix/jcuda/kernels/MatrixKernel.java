package oroarmor.neuralnetwork.matrix.jcuda.kernels;

import oroarmor.neuralnetwork.matrix.jcuda.kernels.functions.AbsKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath.AddKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath.AddValueKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath.MultiplyKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath.MultiplyValueKernel;
import oroarmor.neuralnetwork.util.JCudaKernel;

public abstract class MatrixKernel extends JCudaKernel {

	public MatrixKernel(String name, String subpath) {
		super(name);
		loadKernel("src/data/matrixKernels/" + subpath + "/" + name + ".cu");
	}

	public static void initializeAllKernels() {
		System.out.println("Initializing all kernels...");
		long millis = System.currentTimeMillis();
		AbsKernel.getInstance();
		AddKernel.getInstance();
		AddValueKernel.getInstance();
		MultiplyKernel.getInstance();
		MultiplyValueKernel.getInstance();
		System.out.println("All kernels initialized in " + (System.currentTimeMillis() - millis) + " milliseconds.");
	}

}
