package oroarmor.matrix;

public class SigmoidMatrix extends MatrixFunction {

	public SigmoidMatrix() {
	}

	public Matrix applyFunction(Matrix matrix) {
		Matrix newMatrix = new Matrix(matrix.getRows(), matrix.getCols());

		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				newMatrix.setValue(i, j, sigmoid(matrix.getValue(i, j)));
			}
		}

		return newMatrix;
	}

	public double sigmoid(double value) {
		return 1d / (1d + Math.pow(Math.E, -1d * value));
	}

	public Matrix getDerivative(Matrix matrix) {
		Matrix newMatrix = new Matrix(matrix.getRows(), matrix.getCols());

		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				newMatrix.setValue(i, j, dsigmoid(matrix.getValue(i, j)));
			}
		}
		return newMatrix;
	}

	public double dsigmoid(double value) {
		return value * (1 - value);
	}

}
