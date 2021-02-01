package com.oroarmor.neural_network.matrix.function;

import com.oroarmor.neural_network.matrix.Matrix;

/**
 * An interface that is used to apply values forward and get the derivatives of matrices in layers
 * @author OroArmor
 */
public interface MatrixFunction {
    /**
     * Applies a function to a matrix
     * @param matrix The matrix
     * @param <T> The matrix class
     * @return The matrix with the function applied
     */
    <T extends Matrix<T>> T applyFunction(T matrix);

    /**
     * Gets the derivative of this function on the matrix
     * @param matrix The matrix
     * @param <T> The matrix class
     * @return The matrix as a derivative of this function
     */
    <T extends Matrix<T>> T getDerivative(T matrix);
}
