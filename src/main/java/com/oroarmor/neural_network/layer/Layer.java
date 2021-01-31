package com.oroarmor.neural_network.layer;

import java.io.Serializable;

import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.MatrixFunction;

public abstract class Layer<T extends Matrix<T>> implements Serializable {
    private static final long serialVersionUID = 10L;

    protected int neurons;
    transient protected Matrix.MatrixType type;

    public Layer(int neurons, Matrix.MatrixType type) {
        this.neurons = neurons;
        this.type = type;
    }

    public abstract T feedForward(T inputs);

    public abstract MatrixFunction getMatrixFunction();

    public abstract int getOutputNeurons();

    public abstract T[] getParameters();

    public abstract void setParameters(T[] parameters);

    public abstract T getWeights();

    public abstract void setWeights(T newWeights);

    public abstract void setup(int inputs);

    public abstract Layer<CPUMatrix> convertToCPU();
}
