package com.oroarmor.neural_network.matrix.function;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;

/**
 * A function that keeps only the positive values of a matrix
 * @author OroArmor
 */
@SuppressWarnings("unchecked")
public class KeepPositiveFunction implements MatrixFunction {
    @Override
    public <T extends Matrix<T>> T applyFunction(T matrix) {
        T newMatrix = (T) new CPUMatrix(matrix.getRows(), matrix.getCols());

        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getCols(); j++) {
                newMatrix.setValue(i, j, keepPos(matrix.getValue(i, j)));
            }
        }

        return newMatrix;
    }

    private double dKeepPos(double val) {
        if (val > 0) {
            return 1;
        }
        return 0;
    }

    @Override
    public <T extends Matrix<T>> T getDerivative(T matrix) {
        T newMatrix = (T) new CPUMatrix(matrix.getRows(), matrix.getCols());

        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getCols(); j++) {
                newMatrix.setValue(i, j, dKeepPos(matrix.getValue(i, j)));
            }
        }

        return newMatrix;
    }

    private double keepPos(double val) {
        if (val > 0) {
            return val;
        }
        return 0;
    }
}
