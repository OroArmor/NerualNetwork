package com.oroarmor.neural_network.matrix.function;

import com.oroarmor.neural_network.matrix.Matrix;

/**
 * A function that applies a soft max to the matrix, making the sum 1
 */
public class SoftMaxFunction implements MatrixFunction {
    @Override
    public <T extends Matrix<T>> T applyFunction(T matrix) {
        double total = matrix.sum();
        return matrix.divide(total == 0 ? 1 : total);
    }

    @Override
    public <T extends Matrix<T>> T getDerivative(T matrix) {
        return matrix;
    }
}
