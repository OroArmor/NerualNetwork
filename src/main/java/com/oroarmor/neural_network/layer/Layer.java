package com.oroarmor.neural_network.layer;

import java.io.Serializable;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.MatrixFunction;

/**
 * An abstract implementation for all Layers
 * @param <T> The Matrix class for the layer
 * @author OroArmor
 */
public abstract class Layer<T extends Matrix<T>> implements Serializable {
    private static final long serialVersionUID = 10L;

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
}
