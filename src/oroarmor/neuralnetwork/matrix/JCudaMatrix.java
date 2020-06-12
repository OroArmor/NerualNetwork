package oroarmor.neuralnetwork.matrix;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public class JCudaMatrix extends Matrix {

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;
	private CUdeviceptr deviceMPtr;
	private Pointer matrixPointer, rowPointer, columnPointer, sizePointer;

	public JCudaMatrix(int rows) {
		super(rows);
		createPointers();
	}

	public JCudaMatrix(int rows, int cols) {
		super(rows, cols);
		createPointers();
	}

	public JCudaMatrix(double[] matrixArray, int rows, int cols) {
		super(matrixArray, rows, cols);
		createPointers();
	}

	public void createPointers() {
		int matrixByteSize = rows * cols * Sizeof.DOUBLE;

		deviceMPtr = new CUdeviceptr();
		cuMemAlloc(deviceMPtr, matrixByteSize);
		cuMemcpyHtoD(deviceMPtr, Pointer.to(matrix), matrixByteSize);

		matrixPointer = Pointer.to(deviceMPtr);
		rowPointer = Pointer.to(new int[] { rows });
		columnPointer = Pointer.to(new int[] { cols });
		sizePointer = Pointer.to(new int[] { rows * cols });
	}

	public Pointer getMatrixPointer() {
		return matrixPointer;
	}

	public Pointer getRowPointer() {
		return rowPointer;
	}

	public Pointer getColumnPointer() {
		return columnPointer;
	}

	public Pointer getSizePointer() {
		return sizePointer;
	}

	public Matrix toCPUMatrix() {
		double[] gpuMatrix = new double[rows * cols];
		cuMemcpyDtoH(Pointer.to(gpuMatrix), deviceMPtr, rows * cols * Sizeof.DOUBLE);

		return new Matrix(gpuMatrix, rows, cols);
	}

}
