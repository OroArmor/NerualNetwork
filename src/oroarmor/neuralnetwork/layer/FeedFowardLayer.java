package oroarmor.neuralnetwork.layer;

import java.util.Random;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.MatrixFunction;
import oroarmor.neuralnetwork.matrix.SigmoidMatrix;

public class FeedFowardLayer extends Layer {

	private static final long serialVersionUID = 1L;

	int neurons;
	int previousNeurons;
	Matrix weights;

	public FeedFowardLayer(int neurons) {
		this.neurons = neurons;
	}

	@Override
	public Matrix feedFoward(Matrix inputs) {
		return weights.multiplyMatrix(inputs).applyFunction(new SigmoidMatrix()); // sig(W*I)
	}

	@Override
	public MatrixFunction getMatrixFunction() {
		return new SigmoidMatrix();
	}

	@Override
	public int getOutputNeurons() {
		return neurons;
	}

	@Override
	public Matrix[] getParameters() {
		return new Matrix[] { weights };
	}

	@Override
	public synchronized Matrix getWeights() {
		return weights;
	}

	@Override
	public void setParameters(Matrix[] parameters) {
		weights = parameters[0];
	}

	@Override
	public void setup(int previousNeurons) {
		this.previousNeurons = previousNeurons;

		weights = Matrix.randomMatrix(neurons, previousNeurons, new Random(), -0.1, 0.1);

	}

	@Override
	public synchronized void setWeights(Matrix newWeights) {
		weights = newWeights;
	}

	@Override
	public Matrix backPropagate(Matrix errors) {

		// TODO Auto-generated method stub
		return null;
	}

}
