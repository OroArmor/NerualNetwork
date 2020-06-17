package oroarmor.neuralnetwork.layer;

import java.io.Serializable;

import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.Matrix.MatrixType;
import oroarmor.neuralnetwork.matrix.function.MatrixFunction;

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
