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

package com.oroarmor.neural_network.layer;

import java.util.Random;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.KeepPositiveFunction;
import com.oroarmor.neural_network.matrix.function.MatrixFunction;
import com.oroarmor.neural_network.util.SerializationIndexer;

/**
 * A Keep positive layer. All values less than 0 are converted to 0 by {@link KeepPositiveFunction}
 * @param <T> The Matrix class
 * @author OroArmor
 */
public class KeepPositiveLayer<T extends Matrix<T>> extends Layer<T> {
    private static final long serialVersionUID = SerializationIndexer.KEEP_POSITIVE_LAYER_ID;

    /**
     * A dud field to prevent NPE
     */
    T weights;

    /**
     * Creates a new {@link KeepPositiveLayer}
     * @param neurons The number of output neurons
     * @param type The type of the matrix for the layer
     */
    public KeepPositiveLayer(int neurons, Matrix.MatrixType type) {
        super(neurons, type);
    }

    @Override
    public T feedForward(T inputs) {
        return weights.multiplyMatrix(inputs).applyFunction(new KeepPositiveFunction());
    }

    @Override
    public MatrixFunction getMatrixFunction() {
        return new KeepPositiveFunction();
    }

    @Override
    public int getOutputNeurons() {
        return neurons;
    }

    @Override
    public T getWeights() {
        return weights;
    }

    @Override
    public void setWeights(T newWeights) {
    }

    @Override
    public void setup(int inputs) {
        weights = new CPUMatrix(neurons, inputs).toMatrix(this.type);
    }

    @Override
    public Layer<CPUMatrix> convertToCPU() {
        return new KeepPositiveLayer<>(neurons, Matrix.MatrixType.CPU);
    }

    @Override
    public T backPropagate(T errors) {
        return null;
    }
}
