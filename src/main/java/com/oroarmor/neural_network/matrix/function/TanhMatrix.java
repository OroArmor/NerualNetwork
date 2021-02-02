/*
 * MIT License
 *
 * Copyright (c) 2021 OroArmor
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.oroarmor.neural_network.matrix.function;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;

/**
 * A function that applies the tanh function to a matrix
 * @author OroArmor
 */
@SuppressWarnings("unchecked")
public class TanhMatrix implements MatrixFunction {
    @Override
    public <T extends Matrix<T>> T applyFunction(T matrix) {
        T newMatrix = (T) new CPUMatrix(matrix.getRows(), matrix.getCols());

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
    public <T extends Matrix<T>> T getDerivative(T matrix) {
        T newMatrix = (T) new CPUMatrix(matrix.getRows(), matrix.getCols());

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
