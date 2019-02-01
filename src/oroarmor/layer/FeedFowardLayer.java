package oroarmor.layer;

import java.util.Random;

import oroarmor.matrix.Matrix;
import oroarmor.matrix.MatrixFunction;
import oroarmor.matrix.SigmoidFunction;

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
		Matrix output = weights.multiplyMatrix(inputs);
		output.addMatrix(bias);
		output.applyFunction(new SigmoidFunction());
		
		return output;
	}

	public void correctErrors(Matrix errors, double learningRate) {

	}

	@Override
	public int getOutputNeurons() {
		return neurons;
	}

	@Override
	public Matrix getWeights() {
		return weights;
	}

	@Override
	public void setWeights(Matrix newWeights) {
		weights = newWeights;
	}

	@Override
	public Matrix getBias() {
		return bias;
	}

	@Override
	public void setBias(Matrix newBias) {
		bias= newBias;
	}

	@Override
	public MatrixFunction getMatrixFunction() {
		return new SigmoidFunction();
	}

}
