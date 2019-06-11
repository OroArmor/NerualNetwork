package oroarmor.neuralnetwork.matrix;

public class TanhMatrix extends MatrixFunction {

	public TanhMatrix() {
	}

	@Override
	public Matrix applyFunction(Matrix matrix) {
		Matrix newMatrix = new Matrix(matrix.getRows(), matrix.getCols());

		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				newMatrix.setValue(i, j, tanh(matrix.getValue(i, j)));
			}
		}

		return newMatrix;
	}

	private double dtanh(double value) {
		return 1 - Math.pow(tanh(value), 2);
	}

	@Override
	public Matrix getDerivative(Matrix matrix) {
		Matrix newMatrix = new Matrix(matrix.getRows(), matrix.getCols());

		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				newMatrix.setValue(i, j, dtanh(matrix.getValue(i, j)));
			}
		}

		return newMatrix;
	}

	private double tanh(double value) {
		return (Math.pow(Math.E, 2 * value) - 1) / (Math.pow(Math.E, 2 * value) + 1);
	}

}
