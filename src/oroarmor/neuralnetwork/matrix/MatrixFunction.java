package oroarmor.neuralnetwork.matrix;

public abstract class MatrixFunction {
	public abstract Matrix applyFunction(Matrix matrix);

	public abstract Matrix getDerivative(Matrix matrix);
}
