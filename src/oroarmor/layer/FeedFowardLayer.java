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
	Matrix bias;

	public FeedFowardLayer(int neurons) {
		this.neurons = neurons;
	}

	public void setup(int previousNeurons) {
		this.previousNeurons = previousNeurons;

		this.weights = Matrix.randomMatrix(neurons, previousNeurons, new Random(), -1, 1);

		this.bias = Matrix.randomMatrix(neurons, 1, new Random(), -1, 1);
	}

	public Matrix feedFoward(Matrix inputs) {
		Matrix output = weights.multiplyMatrix(inputs).addMatrix(bias).applyFunction(new SigmoidMatrix());
		return output;
	}

	public void correctErrors(Matrix errors, double learningRate) {

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
		return new Matrix[] {this.weights, this.bias};
	}

	@Override
	public void setParameters(Matrix[] parameters) {
		this.weights = parameters[0];
		this.bias = parameters[1];
	}

}
