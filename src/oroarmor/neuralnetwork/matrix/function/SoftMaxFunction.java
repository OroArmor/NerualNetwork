package oroarmor.neuralnetwork.matrix.function;

import oroarmor.neuralnetwork.matrix.Matrix;

public class SoftMaxFunction extends MatrixFunction {

	double total;

	public SoftMaxFunction() {
	}

	@Override
	public <T extends Matrix<T>> T applyFunction(T matrix) {
		total = matrix.sum();
		return matrix.divide(total);
	}

	@Override
	public <T extends Matrix<T>> T getDerivative(T matrix) {
		return matrix;
	}

	public double getTotal() {
		return total;
	}

}
