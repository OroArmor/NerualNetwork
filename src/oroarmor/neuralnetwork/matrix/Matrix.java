package oroarmor.neuralnetwork.matrix;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Random;

public class Matrix implements Serializable {

	private static final long serialVersionUID = 1L;

	// A couple more constructors
	public static Matrix randomMatrix(int rows, int cols, Random rand, double lowerBound, double upperBound) {
		Matrix randomMatrix = new Matrix(rows, cols);
		randomMatrix.randomize(rand, lowerBound, upperBound);
		return randomMatrix;
	}

	// values
	double[][] matrix;
	int rows;

	int cols;

	public Matrix(double[] array) {
		rows = array.length;
		cols = 1;

		matrix = new double[array.length][1];
		for (int i = 0; i < rows; i++) {
			matrix[i][0] = array[i];
		}
	}

	public Matrix(double[][] array) {
		rows = array.length;
		cols = array[0].length;
		matrix = array;
	}

	public Matrix(int rows) {
		this(rows, 1);
	}

	// constructors
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;

		matrix = new double[rows][cols];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i][j] = 0;
			}
		}
	}

	public Matrix abs() {
		Matrix abs = new Matrix(getRows(), 1);
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				abs.setValue(i, j, Math.abs(getValue(i, j)));
			}
		}
		return abs;
	}

	public Matrix add(double val) {
		Matrix sum = new Matrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				double currentProduct = getValue(i, j) + val;
				sum.setValue(i, j, currentProduct);
			}
		}

		return sum;
	}

	// matrix operations
	public Matrix addMatrix(Matrix other) {

		if (other.rows != rows || other.cols != cols) {
			throw new IllegalArgumentException("Cannot add a " + getRows() + "x" + getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}

		Matrix sum = new Matrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				double currentSum = getValue(i, j) + other.getValue(i, j);
				sum.setValue(i, j, currentSum);
			}
		}
		return sum;
	}

	public Matrix addOnetoEnd() {
		Matrix modified = new Matrix(getRows() + 1, getCols());

		for (int j = 0; j < modified.getCols(); j++) {
			for (int i = 0; i < modified.getRows() - 1; i++) {
				modified.setValue(i, j, getValue(i, j));
			}
			modified.setValue(modified.getRows() - 1, j, 1);
		}

		return modified;
	}

	// functions
	public Matrix applyFunction(MatrixFunction function) {
		return function.applyFunction(this);
	}

	@Override
	public Matrix clone() {
		Matrix duplicate = new Matrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				duplicate.setValue(i, j, getValue(i, j));
			}
		}
		return duplicate;
	}

	public Matrix collapseRows() {
		Matrix collapsed = new Matrix(getRows(), 1);

		for (int i = 0; i < getRows(); i++) {
			double rowSum = 0;
			for (int j = 0; j < getCols(); j++) {
				rowSum += getValue(i, j);
			}
			collapsed.setValue(i, 0, rowSum);
		}

		return collapsed;
	}

	public Matrix divide(double val) {
		if (val == 0) {
			throw new IllegalArgumentException("Argument 'divisor' is 0");
		}
		return multiply(1 / val);
	}

	public int getCols() {
		return cols;
	}

	public Matrix getDerivative(MatrixFunction function) {
		return function.getDerivative(this);
	}

	// gets and sets
	public int getRows() {
		return rows;
	}

	public double getValue(int row, int col) {
		return matrix[row][col];
	}

	public double[][] getValues() {
		return matrix;
	}

	public Matrix hadamard(Matrix other) {
		if (getRows() != other.getRows() || getCols() != other.getCols()) {
			throw new IllegalArgumentException("Cannot multiply a " + getRows() + "x" + getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}
		Matrix product = new Matrix(getRows(), getCols());
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				product.setValue(i, j, getValue(i, j) * other.getValue(i, j));
			}
		}

		return product;
	}

	// value operations
	public Matrix multiply(double val) {
		Matrix product = new Matrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				double currentProduct = getValue(i, j) * val;
				product.setValue(i, j, currentProduct);
			}
		}

		return product;
	}

	public synchronized Matrix multiplyMatrix(Matrix other) {

		if (getCols() != other.getRows()) {
			throw new IllegalArgumentException("Cannot multiply a " + getRows() + "x" + getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}

		Matrix product = new Matrix(getRows(), other.getCols());

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

	public Matrix pow(double power) {
		Matrix duplicate = new Matrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				duplicate.setValue(i, j, Math.pow(getValue(i, j), power));
			}
		}
		return duplicate;
	}

	public Matrix exp() {
		Matrix duplicate = new Matrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				duplicate.setValue(i, j, Math.exp(getValue(i, j)));
			}
		}
		return duplicate;
	}

	// prints
	public Matrix print(String format) {
		DecimalFormat df = new DecimalFormat(format);
		for (int i = 0; i < getRows(); i++) {
			System.out.print("| ");
			for (int j = 0; j < getCols(); j++) {
				System.out.print(df.format(getValue(i, j)) + " ");
			}
			System.out.println(" |");
		}
		System.out.println(" ");
		return this;
	}

	public Matrix print() {
		return print("#.##");
	}

	public void randomize(Random rand, double lowerBound, double upperBound) {
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				setValue(i, j, rand.nextDouble() * (upperBound - lowerBound) + lowerBound);
			}
		}
	}

	public void setValue(int row, int col, double val) {
		matrix[row][col] = val;
	}

	public Matrix subtract(double val) {
		return add(-val);
	}

	public synchronized Matrix subtractMatrix(Matrix other) {
		Matrix sum = new Matrix(getRows(), getCols());

		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				double currentSum = getValue(i, j) - other.getValue(i, j);
				sum.setValue(i, j, currentSum);
			}
		}
		return sum;
	}

	public Matrix transpose() {
		Matrix transposed = new Matrix(getCols(), getRows());
		for (int i = 0; i < getRows(); i++) {
			for (int j = 0; j < getCols(); j++) {
				transposed.setValue(j, i, getValue(i, j));
			}
		}

		return transposed;
	}

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

	public Matrix stack(Matrix other) {
		Matrix stacked = new Matrix(Math.max(getRows(), other.getRows()), getCols() + other.getCols());

		for (int i = 0; i < stacked.getRows(); i++) {
			for (int j = 0; j < stacked.getCols(); j++) {
				if (getCols() > j) {
					if (getRows() < i) {
						stacked.setValue(i, j, 0);
					} else {
						stacked.setValue(i, j, getValue(i, j));
					}
				} else {
					if (other.getRows() < i) {
						stacked.setValue(i, j, 0);
					} else {
						stacked.setValue(i, j, other.getValue(i, j - getCols()));
					}
				}
			}
		}

		return stacked;
	}

	public double getSum() {
		return collapseRows().transpose().collapseRows().getValue(0, 0);
	}
}
