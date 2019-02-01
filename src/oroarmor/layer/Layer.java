package oroarmor.layer;

import java.io.Serializable;

import oroarmor.matrix.Matrix;
import oroarmor.matrix.MatrixFunction;

public abstract class Layer implements Serializable {

	private static final long serialVersionUID = 1L;

	public Layer() {
	}

	public abstract void setup(int inputs);

	public abstract Matrix feedFoward(Matrix inputs);

	public abstract int getOutputNeurons();

	public abstract Matrix getWeights();

	public abstract void setWeights(Matrix newWeights);

	public abstract Matrix getBias();

	public abstract void setBias(Matrix newBias);
	
	public abstract MatrixFunction getMatrixFunction();

}
