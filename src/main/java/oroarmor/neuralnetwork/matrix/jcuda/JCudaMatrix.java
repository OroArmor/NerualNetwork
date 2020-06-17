package oroarmor.neuralnetwork.matrix.jcuda;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import java.util.HashMap;
import java.util.Random;
import java.util.stream.Collectors;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.function.MatrixFunction;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.functions.AbsKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.functions.RandomKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath.AddKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath.AddValueKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath.MultiplyKernel;
import oroarmor.neuralnetwork.matrix.jcuda.kernels.simpleMath.MultiplyValueKernel;

public class JCudaMatrix implements Matrix<JCudaMatrix> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2L;

	private CUdeviceptr deviceMPtr;

	public CUdeviceptr getDeviceMPtr() {
		return deviceMPtr;
	}

	private Pointer matrixPointer, rowPointer, columnPointer, sizePointer;

	private static HashMap<JCudaMatrix, Runnable> disposers = new HashMap<JCudaMatrix, Runnable>();

	// values
	double[] matrix;
	int rows;
	int cols;

	private boolean keep = false;
	private boolean dirty = true;

	public JCudaMatrix(int rows) {
		this(rows, 1);
	}

	public JCudaMatrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		disposers.entrySet().stream().filter(e -> !e.getKey().keep).collect(Collectors.toList())
				.forEach(e -> disposers.remove(e.getKey()).run());
		createPointersNoBackingArray();
		disposers.put(this, () -> this.dispose());
	}

	public JCudaMatrix(double[] matrixArray, int rows, int cols) {
		matrix = matrixArray;
		this.rows = rows;
		this.cols = cols;
		disposers.entrySet().stream().filter(e -> !e.getKey().keep).collect(Collectors.toList())
				.forEach(e -> disposers.remove(e.getKey()).run());
		createPointers();
		disposers.put(this, () -> this.dispose());
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

	public void createPointersNoBackingArray() {
		int matrixByteSize = rows * cols * Sizeof.DOUBLE;

		deviceMPtr = new CUdeviceptr();
		cuMemAlloc(deviceMPtr, matrixByteSize);

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

	public CPUMatrix toCPUMatrix() {
		matrix = new double[rows * cols];
		cuMemcpyDtoH(Pointer.to(matrix), deviceMPtr, rows * cols * Sizeof.DOUBLE);

		return new CPUMatrix(matrix, rows, cols);
	}

	public void dispose() {
		cuMemFree(deviceMPtr);
	}

	public JCudaMatrix keep() {
		keep = true;
		return this;
	}

	@Override
	public JCudaMatrix abs() {
		JCudaMatrix abs = new JCudaMatrix(this.rows, this.cols);
		AbsKernel.getInstance().abs(this, abs);
		dirty = true;
		return abs;
	}

	@Override
	public JCudaMatrix add(double val) {
		JCudaMatrix sum = new JCudaMatrix(this.rows, this.cols);
		AddValueKernel.getInstance().add(this, val, sum);
		dirty = true;
		return sum;
	}

	@Override
	public JCudaMatrix addMatrix(JCudaMatrix other) {
		JCudaMatrix sum = new JCudaMatrix(this.rows, this.cols);
		AddKernel.getInstance().add(this, other, sum);
		dirty = true;
		return sum;
	}

	@Override
	public JCudaMatrix applyFunction(MatrixFunction function) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JCudaMatrix divide(double val) {
		return multiply(1d / val);
	}

	@Override
	public int getCols() {
		return cols;
	}

	@Override
	public JCudaMatrix getDerivative(MatrixFunction function) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getRows() {
		return rows;
	}

	@Override
	public double getValue(int row, int col) {
		if (dirty) {
			dirty = false;
			return this.toCPUMatrix().getValue(row, col);
		}
		return matrix[row * cols + col];
	}

	@Override
	public double[] getValues() {
		if (dirty) {
			dirty = false;
			return this.toCPUMatrix().getValues();
		}
		return matrix;
	}

	@Override
	public double sum() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public JCudaMatrix hadamard(JCudaMatrix other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JCudaMatrix multiply(double val) {
		JCudaMatrix product = new JCudaMatrix(this.rows, this.cols);
		MultiplyValueKernel.getInstance().multiplyValue(this, val, product);
		return product;
	}

	@Override
	public JCudaMatrix multiplyMatrix(JCudaMatrix other) {
		JCudaMatrix product = new JCudaMatrix(rows, other.cols);
		MultiplyKernel.getInstance().multiply(this, other, product);
		return product;
	}

	@Override
	public JCudaMatrix pow(double power) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void randomize(Random rand, double lowerBound, double upperBound) {
		RandomKernel.getInstance().random(this, rand, lowerBound, upperBound);
	}

	@Override
	public void setValue(int row, int col, double val) {
		// TODO Auto-generated method stub

	}

	@Override
	public JCudaMatrix subtract(double val) {
		return add(-val);
	}

	@Override
	public JCudaMatrix subtractMatrix(JCudaMatrix other) {
		return this.addMatrix(other.multiply(-1));
	}

	@Override
	public JCudaMatrix transpose() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getMax() {
		// TODO Auto-generated method stub
		return 0;
	}

	public static JCudaMatrix randomMatrix(int rows, int cols, Random rand, double lowerBound, double upperBound) {
		JCudaMatrix matrix = new JCudaMatrix(rows, cols);
		matrix.randomize(rand, lowerBound, upperBound);
		return matrix;
	}

}
