package oroarmor.neuralnetwork.layer;

import java.io.Serializable;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.MatrixFunction;

public abstract class Layer implements Serializable {

	private static final long serialVersionUID = 1L;

	public Layer() {
	}

	public abstract Matrix feedFoward(Matrix inputs);

	public abstract MatrixFunction getMatrixFunction();

	public abstract int getOutputNeurons();

	public abstract Matrix[] getParameters();

	public abstract Matrix getWeights();

	public abstract void setParameters(Matrix[] parameters);

	public abstract void setup(int inputs);

	public abstract void setWeights(Matrix newWeights);

	public abstract Matrix backPropagate(Matrix errors);
}
