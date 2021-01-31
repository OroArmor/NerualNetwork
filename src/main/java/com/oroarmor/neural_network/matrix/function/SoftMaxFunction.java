package com.oroarmor.neural_network.matrix.function;

import com.oroarmor.neural_network.matrix.Matrix;

public class SoftMaxFunction extends MatrixFunction {
    double total;

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
