package oroarmor.matrix;

import java.io.Serializable;
import java.util.Random;

public class Matrix implements Serializable {

	private static final long serialVersionUID = 1L;

	// values
	double[][] matrix;
	int rows;
	int cols;

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

	public Matrix(int rows) {
		this(rows, 1);
	}

	public Matrix(double[] array) {
		this.rows = array.length;
		this.cols = 1;

		matrix = new double[array.length][1];
		for (int i = 0; i < rows; i++) {
			matrix[i][0] = array[i];
		}
	}

	public Matrix(double[][] array) {
		this.rows = array.length;
		this.cols = array[0].length;
		matrix = array;
	}

	public void randomize(Random rand, double lowerBound, double upperBound) {
		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				this.setValue(i, j, rand.nextDouble() * (upperBound - lowerBound) + lowerBound);
			}
		}
	}

	// A couple more constructors
	public static Matrix randomMatrix(int rows, int cols, Random rand, double lowerBound, double upperBound) {
		Matrix randomMatrix = new Matrix(rows, cols);
		randomMatrix.randomize(rand, lowerBound, upperBound);
		return randomMatrix;
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

	public Matrix abs() {
		Matrix abs = new Matrix(this.getRows(), 1);
		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				abs.setValue(i, j, Math.abs(this.getValue(i, j)));
			}
		}
		return abs;
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

	public Matrix divide(double val) {
		if (val == 0) {
			throw new IllegalArgumentException("Argument 'divisor' is 0");
		}
		return this.multiply(1 / val);
	}

	public Matrix subtract(double val) {
		return this.add(-val);
	}

	// matrix operations
	public Matrix addMatrix(Matrix other) {

		if (other.rows != this.rows || other.cols != this.cols) {
			throw new IllegalArgumentException("Rows and Cols dont line up");
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

	public Matrix subtractMatrix(Matrix other) {
		Matrix sum = new Matrix(this.getRows(), this.getCols());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getCols(); j++) {
				double currentSum = this.getValue(i, j) - other.getValue(i, j);
				sum.setValue(i, j, currentSum);
			}
		}
		return sum;
	}

	public Matrix multiplyMatrix(Matrix other) {

		if (this.getCols() != other.getRows()) {
			throw new IllegalArgumentException("Rows and Columns dont line up");
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

	// functions
	public Matrix applyFunction(MatrixFunction function) {
		return function.applyFunction(this);
	}

	public Matrix getDerivative(MatrixFunction function) {
		return function.getDerivative(this);
	}

	// gets and sets
	public int getRows() {
		return rows;
	}

	public int getCols() {
		return cols;
	}

	public double getValue(int row, int col) {
		return matrix[row][col];
	}

	public void setValue(int row, int col, double val) {
		matrix[row][col] = val;
	}

	// prints
	public void print() {
		for (int i = 0; i < this.getRows(); i++) {
			System.out.print("| ");
			for (int j = 0; j < this.getCols(); j++) {
				System.out.print(this.getValue(i, j) + " ");
			}
			System.out.println(" |");
		}
		System.out.println(" ");
	}
}
