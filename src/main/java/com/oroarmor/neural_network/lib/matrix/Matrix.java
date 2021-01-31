package com.oroarmor.neural_network.lib.matrix;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Random;

import com.oroarmor.neural_network.lib.matrix.function.MatrixFunction;
import com.oroarmor.neural_network.lib.matrix.jcuda.JCudaMatrix;

public interface Matrix<T extends Matrix<T>> extends Serializable {

	T abs();

	T add(double val);

	// matrix operations
	T addMatrix(T other);

	// functions
	T applyFunction(MatrixFunction function);

	T divide(double val);

	int getCols();

	T getDerivative(MatrixFunction function);

	// gets and sets
	int getRows();

	double getValue(int row, int col);

	double[] getValues();

	double sum();

	T hadamard(T other);

	// value operations
	T multiply(double val);

	T multiplyMatrix(T other);

	T pow(double power);

	@SuppressWarnings("unchecked")
	default T print(String format) {
		DecimalFormat df = new DecimalFormat(format);

		for (int i = 0; i < getRows(); i++) {
			System.out.print("| ");
			for (int j = 0; j < getCols(); j++) {
				System.out.print(df.format(getValue(i, j)) + " ");
			}
			System.out.println("|");
		}
		System.out.println(" ");
		return (T) this;
	}

	default T print() {
		return print("#.##");
	}

	void randomize(Random rand, double lowerBound, double upperBound);

	void setValue(int row, int col, double val);

	T subtract(double val);

	T subtractMatrix(T other);

	T transpose();

	int getMax();

	@SuppressWarnings("unchecked")
	default <S extends Matrix<S>> S toMatrix(MatrixType type) {
		switch (type) {
			case CPU:
				if (this instanceof CPUMatrix)
					return (S) this;
				if (this instanceof JCudaMatrix)
					return (S) ((JCudaMatrix) this).toCPUMatrix();
				break;
			case JCuda:
				if (this instanceof JCudaMatrix)
					return (S) this;
				if (this instanceof CPUMatrix)
					return (S) (new JCudaMatrix(this.getValues(), this.getRows(), this.getCols()));
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	static <T extends Matrix<T>> T randomMatrix(MatrixType type, int rows, int cols, Random rand,
			double lowerBound, double upperBound) {
		switch (type) {
			case CPU:
				return (T) CPUMatrix.randomMatrix(rows, cols, rand, lowerBound, upperBound);
			case JCuda:
				return (T) JCudaMatrix.randomMatrix(rows, cols, rand, lowerBound, upperBound);
		}
		return null;
	}

	enum MatrixType {
		CPU, JCuda
	}

}