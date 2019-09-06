package oroarmor.neuralnetwork.matrix;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

public class GMatrix extends Matrix {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	Pointer rowP, colP, matrixP, sizeP;

	public GMatrix(int rows) {
		super(rows);
		createPointers();
	}

	public GMatrix(int rows, int cols) {
		super(rows, cols);
		createPointers();
	}

	public GMatrix(double[] matrixArray, int rows, int cols) {
		super(matrixArray, rows, cols);
		createPointers();
	}

	public void createPointers() {
		CUdeviceptr deviceMPtr = new CUdeviceptr();
		cuMemAlloc(deviceMPtr, rows * cols * Sizeof.DOUBLE);
		matrixP = Pointer.to(matrix);
		cuMemcpyHtoD(deviceMPtr, matrixP, rows * cols * Sizeof.DOUBLE);

		CUdeviceptr deviceRPtr = new CUdeviceptr();
		cuMemAlloc(deviceRPtr, Sizeof.INT);
		rowP = Pointer.to(new int[] { rows });
		cuMemcpyHtoD(deviceRPtr, matrixP, Sizeof.INT);

		CUdeviceptr deviceCPtr = new CUdeviceptr();
		cuMemAlloc(deviceCPtr, Sizeof.INT);
		colP = Pointer.to(new int[] { cols });
		cuMemcpyHtoD(deviceCPtr, colP, Sizeof.INT);

		CUdeviceptr deviceSPtr = new CUdeviceptr();
		cuMemAlloc(deviceSPtr, Sizeof.INT);
		sizeP = Pointer.to(new int[] { cols * rows });
		cuMemcpyHtoD(deviceSPtr, sizeP, Sizeof.INT);
	}

	public Pointer getMPointer() {
		return matrixP;
	}

	public Pointer getRPointer() {
		return rowP;
	}

	public Pointer getCPointer() {
		return colP;
	}

	public Pointer getSPointer() {
		return sizeP;
	}

}
