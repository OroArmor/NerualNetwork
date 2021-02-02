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
import com.oroarmor.neural_network.matrix.function.MatrixFunction;
import com.oroarmor.neural_network.matrix.function.SigmoidMatrix;
import com.oroarmor.neural_network.util.SerializationIndexer;

/**
 * A Neural Network layer that feeds the inputs forward through a {@link SigmoidMatrix} function
 * @param <T> The type of Matrix
 * @author OroArmor
 */
public class FeedForwardLayer<T extends Matrix<T>> extends Layer<T> {
    private static final long serialVersionUID = SerializationIndexer.FEED_FORWARD_LAYER_ID;

    /**
     * The previous neurons (inputs)
     */
    protected int previousNeurons;

    /**
     * The weights for the layer (inputs by outputs)
     */
    protected T weights;

    /**
     * Creates a new {@link FeedForwardLayer}
     * @param neurons The number of output neurons
     * @param type The type of matrix for the layer
     */
    public FeedForwardLayer(int neurons, Matrix.MatrixType type) {
        super(neurons, type);
    }

    @Override
    public T feedForward(T inputs) {
        return weights.multiplyMatrix(inputs).applyFunction(new SigmoidMatrix()); // sig(W*I)
    }

    @Override
    public MatrixFunction getMatrixFunction() {
        return new SigmoidMatrix();
    }

    @Override
    public int getOutputNeurons() {
        return neurons;
    }

    @Override
    public synchronized T getWeights() {
        return weights;
    }

    @Override
    public synchronized void setWeights(T newWeights) {
        weights = newWeights;
    }

    @Override
    public void setup(int previousNeurons) {
        this.previousNeurons = previousNeurons;
        weights = Matrix.randomMatrix(type, neurons, previousNeurons, new Random(), -1, 1);
    }

    @Override
    public Layer<CPUMatrix> convertToCPU() {
        FeedForwardLayer<CPUMatrix> newLayer = new FeedForwardLayer<>(neurons, Matrix.MatrixType.CPU);
        newLayer.previousNeurons = previousNeurons;
        newLayer.weights = weights.toMatrix(Matrix.MatrixType.CPU);
        return newLayer;
    }

    @Override
    public T backPropagate(T errors) {
        return null;
    }
}
