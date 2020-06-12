package oroarmor.neuralnetwork.matrix.function;

import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.Matrix;

@SuppressWarnings("unchecked")
public class SigmoidMatrix extends MatrixFunction {

	public SigmoidMatrix() {
	}

	@Override
	public <T extends Matrix<T>> T applyFunction(T matrix) {
		T newMatrix = (T) new CPUMatrix(matrix.getRows(), matrix.getCols());

		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				newMatrix.setValue(i, j, sigmoid(matrix.getValue(i, j)));
			}
		}

		return newMatrix;
	}

	public double dsigmoid(double value) {
		return value * (1 - value);
	}

	@Override
	public <T extends Matrix<T>> T getDerivative(T matrix) {
		T newMatrix = (T) new CPUMatrix(matrix.getRows(), matrix.getCols());

		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				newMatrix.setValue(i, j, dsigmoid(matrix.getValue(i, j)));
			}
		}
		return newMatrix;
	}

	public double sigmoid(double value) {
		return 1d / (1d + Math.pow(Math.E, -1d * value));
	}

}
