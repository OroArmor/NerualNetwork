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

		this.weights = Matrix.randomMatrix(previousNeurons+1, neurons, new Random(), -1, 1);

	}

	public Matrix feedFoward(Matrix inputs) {
		Matrix output = inputs.clone().addOnetoEnd(); // clone the inputs and add one to the end for bias
		output = output.multiplyMatrix(weights); // multiply by the weights and bias to get the unweighted output
		return output.applyFunction(new SigmoidMatrix()); // apply sigmoid and return
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
		return new Matrix[] { this.weights};
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
