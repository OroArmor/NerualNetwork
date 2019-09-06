package oroarmor.numberID;

import jcuda.Pointer;
import oroarmor.neuralnetwork.matrix.GMatrix;
import oroarmor.neuralnetwork.matrix.MatrixKernel;
import oroarmor.neuralnetwork.util.Dim3;

public class NumberIDJCuda {

	public static void main(String[] args) {
		
		
		
		MatrixKernel.InitJCuda(true);
		
		MatrixKernel test = new MatrixKernel("test");
		
		test.loadKernel("src/data/matrixKernels/test.cu");
		
		GMatrix testM = new GMatrix(128,128);
		
		Dim3 blockSize = new Dim3(32);
		Dim3 gridSize = new Dim3(testM.getCols()*testM.getRows()/blockSize.x);
		
		test.runKernel(Pointer.to(testM.getSPointer(), testM.getMPointer()), gridSize, blockSize);
	}

}
