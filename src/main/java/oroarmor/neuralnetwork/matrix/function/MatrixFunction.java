package oroarmor.neuralnetwork.matrix.function;

import oroarmor.neuralnetwork.matrix.Matrix;

public abstract class MatrixFunction {
	public abstract <T extends Matrix<T>> T applyFunction(T matrix);

	public abstract <T extends Matrix<T>> T getDerivative(T matrix);
}
