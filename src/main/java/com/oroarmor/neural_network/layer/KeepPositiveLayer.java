package com.oroarmor.neural_network.layer;

import java.util.Random;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.KeepPositiveFunction;
import com.oroarmor.neural_network.matrix.function.MatrixFunction;

@SuppressWarnings("unchecked")
public class KeepPositiveLayer<T extends Matrix<T>> extends Layer<T> {
    private static final long serialVersionUID = 11L;

    public T weights;

    public KeepPositiveLayer(int neurons, Matrix.MatrixType type) {
        super(neurons, type);
    }

    @Override
    public T feedForward(T inputs) {
        return inputs.applyFunction(new KeepPositiveFunction());
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
    public T[] getParameters() {
        return null;
    }

    @Override
    public void setParameters(T[] parameters) {
    }

    @Override
    public T getWeights() {
        return weights;
    }

    @Override
    public void setWeights(T newWeights) {
        weights = newWeights;
    }

    @Override
    public void setup(int inputs) {
        weights = (T) CPUMatrix.randomMatrix(neurons, inputs, new Random(), -1, 1);
    }

    @Override
    public Layer<CPUMatrix> convertToCPU() {
        KeepPositiveLayer<CPUMatrix> newLayer = new KeepPositiveLayer<>(neurons, Matrix.MatrixType.CPU);
        newLayer.weights = weights.toMatrix(Matrix.MatrixType.CPU);

        return newLayer;
    }
}
