package com.oroarmor.neural_network.lib.matrix.function;

import com.oroarmor.neural_network.lib.matrix.Matrix;

public abstract class MatrixFunction {
	public abstract <T extends Matrix<T>> T applyFunction(T matrix);

	public abstract <T extends Matrix<T>> T getDerivative(T matrix);
}
