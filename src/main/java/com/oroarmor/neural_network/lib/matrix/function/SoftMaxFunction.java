package com.oroarmor.neural_network.lib.matrix.function;

import com.oroarmor.neural_network.lib.matrix.Matrix;

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
