package oroarmor.layer;

import java.util.Random;

import oroarmor.matrix.Matrix;
import oroarmor.matrix.MatrixFunction;
import oroarmor.matrix.SigmoidMatrix;

public class FeedFowardLayer extends Layer {

	private static final long serialVersionUID = 1L;

	int neurons;
	int previousNeurons;
	Matrix weights;

	public FeedFowardLayer(int neurons) {
		this.neurons = neurons;
	}

	public void setup(int previousNeurons) {
		this.previousNeurons = previousNeurons;

		this.weights = Matrix.randomMatrix(neurons, previousNeurons, new Random(), -1, 1);

	}

	public Matrix feedFoward(Matrix inputs) {
		return weights.multiplyMatrix(inputs).applyFunction(new SigmoidMatrix()); // sig(W*I)
	}

	@Override
	public int getOutputNeurons() {
		return neurons;
	}

	@Override
	public MatrixFunction getMatrixFunction() {
		return new SigmoidMatrix();
	}

	@Override
	public Matrix[] getParameters() {
		return new Matrix[] { this.weights };
	}

	@Override
	public void setParameters(Matrix[] parameters) {
		this.weights = parameters[0];
	}

	@Override
	public Matrix getWeights() {
		return this.weights;
	}

	@Override
	public void setWeights(Matrix newWeights) {
		this.weights = newWeights;
	}

}
