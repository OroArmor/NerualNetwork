package oroarmor.neuralnetwork.matrix;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public class JCudaMatrix extends Matrix {

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;
	Pointer rowPointer, colPointer, matrixPointer, sizePointer;

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
		CUdeviceptr deviceMPtr = new CUdeviceptr();
		cuMemAlloc(deviceMPtr, rows * cols * Sizeof.DOUBLE);
		matrixPointer = Pointer.to(matrix);
		cuMemcpyHtoD(deviceMPtr, matrixPointer, rows * cols * Sizeof.DOUBLE);

		CUdeviceptr deviceRPtr = new CUdeviceptr();
		cuMemAlloc(deviceRPtr, Sizeof.INT);
		rowPointer = Pointer.to(new int[] { rows });
		cuMemcpyHtoD(deviceRPtr, rowPointer, Sizeof.INT);

		CUdeviceptr deviceCPtr = new CUdeviceptr();
		cuMemAlloc(deviceCPtr, Sizeof.INT);
		colPointer = Pointer.to(new int[] { cols });
		cuMemcpyHtoD(deviceCPtr, colPointer, Sizeof.INT);

		CUdeviceptr deviceSPtr = new CUdeviceptr();
		cuMemAlloc(deviceSPtr, Sizeof.INT);
		sizePointer = Pointer.to(new int[] { cols * rows });
		cuMemcpyHtoD(deviceSPtr, sizePointer, Sizeof.INT);
	}

	public Pointer getMatrixPointer() {
		return matrixPointer;
	}

	public Pointer getRowPointer() {
		return rowPointer;
	}

	public Pointer getColumnPointer() {
		return colPointer;
	}

	public Pointer getSizePointer() {
		return sizePointer;
	}

}
