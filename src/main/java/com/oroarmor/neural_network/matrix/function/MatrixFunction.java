package com.oroarmor.neural_network.matrix.function;

import com.oroarmor.neural_network.matrix.Matrix;

public abstract class MatrixFunction {
    public abstract <T extends Matrix<T>> T applyFunction(T matrix);

    public abstract <T extends Matrix<T>> T getDerivative(T matrix);
}
