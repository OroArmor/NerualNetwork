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

import java.io.Serializable;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.MatrixFunction;
import com.oroarmor.neural_network.util.SerializationIndexer;

/**
 * An abstract implementation for all Layers
 * @param <T> The Matrix class for the layer
 * @author OroArmor
 */
public abstract class Layer<T extends Matrix<T>> implements Serializable {
    private static final long serialVersionUID = SerializationIndexer.LAYER_ID_HEADER;

    /**
     * The output neurons
     */
    protected int neurons;

    /**
     * The matrix type for the {@link Layer}
     */
    protected Matrix.MatrixType type;

    /**
     * Creates a new {@link Layer}
     * @param neurons The number of output neurons
     * @param type The type of the matrix for the layer
     */
    public Layer(int neurons, Matrix.MatrixType type) {
        this.neurons = neurons;
        this.type = type;
    }

    /**
     * Feeds the inputs through the layer
     * @param inputs The matrix to input
     * @return The output for the layer
     */
    public abstract T feedForward(T inputs);

    /**
     *
     * @return The {@link MatrixFunction} for the layer
     */
    public abstract MatrixFunction getMatrixFunction();

    /**
     *
     * @return The number of output neurons
     */
    public abstract int getOutputNeurons();

    /**
     *
     * @return The weights for the layer
     */
    public abstract T getWeights();

    /**
     * Sets the weights for the layer
     * @param newWeights The new weights
     */
    public abstract void setWeights(T newWeights);

    /**
     * Sets up the layer, should be called once
     * @param inputs The number of inputs into this layer
     */
    public abstract void setup(int inputs);

    /**
     * Converts all matrices into their CPU form
     * @return A new layer for CPU only calculation
     */
    public abstract Layer<CPUMatrix> convertToCPU();

    public abstract T backPropagate(T errors);
}
