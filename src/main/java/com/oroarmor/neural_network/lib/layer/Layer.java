package com.oroarmor.neural_network.lib.layer;

import java.io.Serializable;

import com.oroarmor.neural_network.lib.matrix.CPUMatrix;
import com.oroarmor.neural_network.lib.matrix.Matrix;
import com.oroarmor.neural_network.lib.matrix.Matrix.MatrixType;
import com.oroarmor.neural_network.lib.matrix.function.MatrixFunction;

public abstract class Layer<T extends Matrix<T>> implements Serializable {

	private static final long serialVersionUID = 10L;
	protected int neurons;
	transient protected MatrixType type;

	public Layer(int neurons, MatrixType type) {
		this.neurons = neurons;
		this.type = type;
	}

	public abstract T feedFoward(T inputs);

	public abstract MatrixFunction getMatrixFunction();

	public abstract int getOutputNeurons();

	public abstract T[] getParameters();

	public abstract T getWeights();

	public abstract void setParameters(T[] parameters);

	public abstract void setup(int inputs);

	public abstract void setWeights(T newWeights);

	public abstract Layer<CPUMatrix> contvertToCPU();
}
