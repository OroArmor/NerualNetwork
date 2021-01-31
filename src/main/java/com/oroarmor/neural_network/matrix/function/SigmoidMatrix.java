package com.oroarmor.neural_network.matrix.function;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;

@SuppressWarnings("unchecked")
public class SigmoidMatrix extends MatrixFunction {
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

    public double dSigmoid(double value) {
        return value * (1 - value);
    }

    @Override
    public <T extends Matrix<T>> T getDerivative(T matrix) {
        T newMatrix = (T) new CPUMatrix(matrix.getRows(), matrix.getCols());

        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getCols(); j++) {
                newMatrix.setValue(i, j, dSigmoid(matrix.getValue(i, j)));
            }
        }
        return newMatrix;
    }

    public double sigmoid(double value) {
        return 1d / (1d + Math.pow(Math.E, -1d * value));
    }
}
