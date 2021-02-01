package com.oroarmor.neural_network.layer;

import java.util.Random;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.MatrixFunction;
import com.oroarmor.neural_network.matrix.function.SigmoidMatrix;

/**
 * A Neural Network layer that feeds the inputs forward through a {@link SigmoidMatrix} function
 * @param <T> The type of Matrix
 * @author OroArmor
 */
public class FeedForwardLayer<T extends Matrix<T>> extends Layer<T> {
    private static final long serialVersionUID = 12L;

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
}
