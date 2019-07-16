package oroarmor.neuralnetwork.matrix;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Random;

public class Matrix implements Serializable {

	// col | col
	// row | 0 1
	// row | 2 3
	// row | 4 5
	//

	private static final long serialVersionUID = 1L;

	// A couple more constructors
	public static Matrix randomMatrix(int rows, int cols, Random rand, double lowerBound, double upperBound) {
		Matrix randomMatrix = new Matrix(rows, cols);
		randomMatrix.randomize(rand, lowerBound, upperBound);
		return randomMatrix;
	}

	// values
	double[] matrix;
	int rows;
	int cols;

	public Matrix(int rows) {
		this(rows, 1);
	}

	// constructors
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;

		matrix = new double[rows * cols];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				matrix[i * cols + j] = 0;
			}
		}
	}

	public Matrix(double[] matrixArray, int rows, int cols) {
		this.matrix = matrixArray;
		this.cols = cols;
		this.rows = rows;
	}

	public Matrix abs() {
		Matrix abs = new Matrix(this.getRows(), 1);
		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				abs.setValue(i, j, Math.abs(this.getValue(i, j)));
			}
		}
		return abs;
	}

	public Matrix add(double val) {
		Matrix sum = new Matrix(this.getRows(), this.getCols());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				double currentProduct = this.getValue(i, j) + val;
				sum.setValue(i, j, currentProduct);
			}
		}

		return sum;
	}

	// matrix operations
	public Matrix addMatrix(Matrix other) {

		if (other.rows != this.rows || other.cols != this.cols) {
			throw new IllegalArgumentException("Cannot add a " + this.getRows() + "x" + this.getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}

		Matrix sum = new Matrix(this.getRows(), this.getCols());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				double currentSum = this.getValue(i, j) + other.getValue(i, j);
				sum.setValue(i, j, currentSum);
			}
		}
		return sum;
	}

	public Matrix addOnetoEnd() {
		Matrix modified = new Matrix(this.getRows() + 1, this.getCols());

		for (int j = 0; j < modified.getCols(); j++) {
			for (int i = 0; i < modified.getRows() - 1; i++) {
				modified.setValue(i, j, this.getValue(i, j));
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
		Matrix duplicate = new Matrix(this.getRows(), this.getCols());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				duplicate.setValue(i, j, this.getValue(i, j));
			}
		}
		return duplicate;
	}

	public Matrix collapseRows() {
		Matrix collapsed = new Matrix(this.getRows(), 1);

		for (int i = 0; i < this.getRows(); i++) {
			double rowSum = 0;
			for (int j = 0; j < this.getCols(); j++) {
				rowSum += this.getValue(i, j);
			}
			collapsed.setValue(i, 0, rowSum);
		}

		return collapsed;
	}

	public Matrix divide(double val) {
		if (val == 0) {
			throw new IllegalArgumentException("Argument 'divisor' is 0");
		}
		return this.multiply(1 / val);
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
		return matrix[row * this.getCols() + col];
	}

	public double[] getValues() {
		return matrix;
	}

	public Matrix hadamard(Matrix other) {
		if (this.getRows() != other.getRows() || this.getCols() != other.getCols()) {
			throw new IllegalArgumentException("Cannot multiply a " + this.getRows() + "x" + this.getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}
		Matrix product = new Matrix(this.getRows(), this.getCols());
		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				product.setValue(i, j, this.getValue(i, j) * other.getValue(i, j));
			}
		}

		return product;
	}

	// value operations
	public Matrix multiply(double val) {
		Matrix product = new Matrix(this.getRows(), this.getCols());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				double currentProduct = this.getValue(i, j) * val;
				product.setValue(i, j, currentProduct);
			}
		}

		return product;
	}

	public synchronized Matrix multiplyMatrix(Matrix other) {

		if (this.getCols() != other.getRows()) {
			throw new IllegalArgumentException("Cannot multiply a " + this.getRows() + "x" + this.getCols() + " and a "
					+ other.getRows() + "x" + other.getCols() + " matrix together");
		}

		Matrix product = new Matrix(this.getRows(), other.getCols());

		for (int i = 0; i < product.getRows(); i++) {
			for (int j = 0; j < product.getCols(); j++) {
				double currentVal = 0;

				for (int k = 0; k < this.getCols(); k++) {
					currentVal += this.getValue(i, k) * other.getValue(k, j);
				}

				product.setValue(i, j, currentVal);
			}
		}

		return product;
	}

	public Matrix pow(double power) {
		Matrix duplicate = new Matrix(this.getRows(), this.getCols());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				duplicate.setValue(i, j, Math.pow(this.getValue(i, j), power));
			}
		}
		return duplicate;
	}

	// prints
	public Matrix print(String format) {
		DecimalFormat df = new DecimalFormat(format);
		for (int i = 0; i < this.getRows(); i++) {
			System.out.print("| ");
			for (int j = 0; j < this.getCols(); j++) {
				System.out.print(df.format(this.getValue(i, j)) + " ");
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
		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				this.setValue(i, j, rand.nextDouble() * (upperBound - lowerBound) + lowerBound);
			}
		}
	}

	public void setValue(int row, int col, double val) {
		matrix[row * this.getCols() + col] = val;
	}

	public Matrix subtract(double val) {
		return this.add(-val);
	}

	public synchronized Matrix subtractMatrix(Matrix other) {
		Matrix sum = new Matrix(this.getRows(), this.getCols());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				double currentSum = this.getValue(i, j) - other.getValue(i, j);
				sum.setValue(i, j, currentSum);
			}
		}
		return sum;
	}

	public Matrix transpose() {
		Matrix transposed = new Matrix(this.getCols(), this.getRows());
		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				transposed.setValue(j, i, this.getValue(i, j));
			}
		}

		return transposed;
	}

	public int getMax() {
		int maxIndex = 0;
		double max = Double.MIN_VALUE;

		for (int i = 0; i < this.getRows(); i++) {
			if (this.getValue(i, 0) > max) {
				maxIndex = i;
				max = this.getValue(i, 0);
			}
		}

		return maxIndex;
	}
}
