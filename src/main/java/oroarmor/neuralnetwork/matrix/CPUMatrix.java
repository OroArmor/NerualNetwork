package oroarmor.neuralnetwork.matrix;

import java.util.Random;

import oroarmor.neuralnetwork.matrix.function.MatrixFunction;

public class CPUMatrix implements Matrix<CPUMatrix> {

	// col | col
	// row | 0 1
	// row | 2 3
	// row | 4 5
	//

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	// A couple more constructors
	public static CPUMatrix randomMatrix(int rows, int cols, Random rand, double lowerBound, double upperBound) {
		CPUMatrix randomMatrix = new CPUMatrix(rows, cols);
		randomMatrix.randomize(rand, lowerBound, upperBound);
		return randomMatrix;
	}

	// values
	double[] matrix;
	int rows;
	int cols;

	public CPUMatrix(int rows) {
		this(rows, 1);
	}

	// constructors
	public CPUMatrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;

		matrix = new double[rows * cols];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i * cols + j] = 0;
			}
		}
	}

	public CPUMatrix(double[] matrixArray, int rows, int cols) {
		matrix = matrixArray;
		this.cols = cols;
		this.rows = rows;
	}

	@Override
	public CPUMatrix abs() {
		CPUMatrix abs = new CPUMatrix(getRows(), 1);
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				abs.setValue(i, j, Math.abs(getValue(i, j)));
			}
		}
		return abs;
	}

	@Override
	public CPUMatrix add(double val) {
		CPUMatrix sum = new CPUMatrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				double currentProduct = getValue(i, j) + val;
				sum.setValue(i, j, currentProduct);
			}
		}

		return sum;
	}

	// matrix operations
	@Override
	public CPUMatrix addMatrix(CPUMatrix other) {

		if (other.rows != rows || other.cols != cols) {
			throw new IllegalArgumentException("Cannot add a " + getRows() + "x" + getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}

		CPUMatrix sum = new CPUMatrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				double currentSum = getValue(i, j) + other.getValue(i, j);
				sum.setValue(i, j, currentSum);
			}
		}
		return sum;
	}

	// functions
	@Override
	public CPUMatrix applyFunction(MatrixFunction function) {
		return function.applyFunction(this);
	}

	@Override
	public CPUMatrix divide(double val) {
		if (val == 0) {
			throw new IllegalArgumentException("Argument 'divisor' is 0");
		}
		return multiply(1 / val);
	}

	@Override
	public int getCols() {
		return cols;
	}

	@Override
	public CPUMatrix getDerivative(MatrixFunction function) {
		return function.getDerivative(this);
	}

	// gets and sets
	@Override
	public int getRows() {
		return rows;
	}

	@Override
	public double getValue(int row, int col) {
		return matrix[row * getCols() + col];
	}

	@Override
	public double[] getValues() {
		return matrix;
	}

	@Override
	public CPUMatrix hadamard(CPUMatrix other) {
		if (getRows() != other.getRows() || getCols() != other.getCols()) {
			throw new IllegalArgumentException("Cannot multiply a " + getRows() + "x" + getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}
		CPUMatrix product = new CPUMatrix(getRows(), getCols());
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				product.setValue(i, j, getValue(i, j) * other.getValue(i, j));
			}
		}

		return product;
	}

	// value operations
	@Override
	public CPUMatrix multiply(double val) {
		CPUMatrix product = new CPUMatrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				double currentProduct = getValue(i, j) * val;
				product.setValue(i, j, currentProduct);
			}
		}

		return product;
	}

	@Override
	public synchronized CPUMatrix multiplyMatrix(CPUMatrix other) {

		if (getCols() != other.getRows()) {
			throw new IllegalArgumentException("Cannot multiply a " + getRows() + "x" + getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}

		CPUMatrix product = new CPUMatrix(getRows(), other.getCols());

		for (int i = 0; i < product.getRows(); i++) {
			for (int j = 0; j < product.getCols(); j++) {
				double currentVal = 0;

				for (int k = 0; k < getCols(); k++) {
					currentVal += getValue(i, k) * other.getValue(k, j);
				}

				product.setValue(i, j, currentVal);
			}
		}

		return product;
	}

	@Override
	public CPUMatrix pow(double power) {
		CPUMatrix duplicate = new CPUMatrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				duplicate.setValue(i, j, Math.pow(getValue(i, j), power));
			}
		}
		return duplicate;
	}

	@Override
	public void randomize(Random rand, double lowerBound, double upperBound) {
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				setValue(i, j, rand.nextDouble() * (upperBound - lowerBound) + lowerBound);
			}
		}
	}

	@Override
	public void setValue(int row, int col, double val) {
		matrix[row * getCols() + col] = val;
	}

	@Override
	public CPUMatrix subtract(double val) {
		return add(-val);
	}

	@Override
	public CPUMatrix subtractMatrix(CPUMatrix other) {
		CPUMatrix sum = new CPUMatrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				double currentSum = getValue(i, j) - other.getValue(i, j);
				sum.setValue(i, j, currentSum);
			}
		}
		return sum;
	}

	@Override
	public CPUMatrix transpose() {
		CPUMatrix transposed = new CPUMatrix(getCols(), getRows());
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				transposed.setValue(j, i, getValue(i, j));
			}
		}

		return transposed;
	}

	@Override
	public int getMax() {
		int maxIndex = 0;
		double max = Double.MIN_VALUE;

		for (int i = 0; i < getRows(); i++) {
			if (getValue(i, 0) > max) {
				maxIndex = i;
				max = getValue(i, 0);
			}
		}

		return maxIndex;
	}

	@Override
	public double sum() {
		double sum = 0;
		for (double d : matrix) {
			sum += d;
		}
		return sum;
	}

}
