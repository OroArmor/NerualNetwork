package oroarmor.neuralnetwork.matrix;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Random;

import oroarmor.neuralnetwork.matrix.function.MatrixFunction;

public interface Matrix<T extends Matrix<T>> extends Serializable {

	T abs();

	T add(double val);

	// matrix operations
	T addMatrix(T other);

	T addOnetoEnd();

	// functions
	T applyFunction(MatrixFunction function);

	T clone();

	T collapseRows();

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
	public default T print(String format) {
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

	public default T print() {
		return print("#.##");
	}

	void randomize(Random rand, double lowerBound, double upperBound);

	void setValue(int row, int col, double val);

	T subtract(double val);

	T subtractMatrix(T other);

	T transpose();

	int getMax();

}