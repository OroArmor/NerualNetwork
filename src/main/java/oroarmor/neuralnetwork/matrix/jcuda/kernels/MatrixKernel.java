package oroarmor.neuralnetwork.matrix.jcuda.kernels;

import oroarmor.neuralnetwork.util.JCudaKernel;

public abstract class MatrixKernel extends JCudaKernel {

	public MatrixKernel(String name, String subpath) {
		super(name);
		loadKernel("src/data/matrixKernels/" + subpath + "/" + name + ".cu");
	}

}
