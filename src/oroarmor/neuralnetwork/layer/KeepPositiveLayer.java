package oroarmor.neuralnetwork.layer;

import java.util.Random;

import oroarmor.neuralnetwork.matrix.KeepPositiveFunction;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.MatrixFunction;

public class KeepPositiveLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public Matrix weights;
	int neurons;

	public KeepPositiveLayer(int neurons) {
		this.neurons = neurons;
	}

	@Override
	public Matrix feedFoward(Matrix inputs) {
		return inputs.applyFunction(new KeepPositiveFunction());
	}

	@Override
	public MatrixFunction getMatrixFunction() {
		// TODO Auto-generated method stub
		return new KeepPositiveFunction();
	}

	@Override
	public int getOutputNeurons() {
		return neurons;
	}

	@Override
	public Matrix[] getParameters() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix getWeights() {
		// TODO Auto-generated method stub
		return weights;
	}

	@Override
	public void setParameters(Matrix[] parameters) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setup(int inputs) {
		neurons = inputs;

		weights = Matrix.randomMatrix(neurons, inputs, new Random(), -1, 1);
	}

	@Override
	public void setWeights(Matrix newWeights) {
		weights = newWeights;
	}

	@Override
	public Matrix backPropagate(Matrix errors) {

		return null;
	}

}
