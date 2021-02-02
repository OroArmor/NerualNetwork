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
