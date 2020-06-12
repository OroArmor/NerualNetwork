package oroarmor.neuralnetwork.layer;

import java.util.Random;

import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.function.KeepPositiveFunction;
import oroarmor.neuralnetwork.matrix.function.MatrixFunction;

@SuppressWarnings("unchecked")
public class KeepPositiveLayer<T extends Matrix<T>> extends Layer<T> {

	/**
	 *
	 */
	private static final long serialVersionUID = 11L;
	public T weights;
	int neurons;

	public KeepPositiveLayer(int neurons) {
		this.neurons = neurons;
	}

	@Override
	public T feedFoward(T inputs) {
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
	public T[] getParameters() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public T getWeights() {
		// TODO Auto-generated method stub
		return weights;
	}

	@Override
	public void setParameters(T[] parameters) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setup(int inputs) {
		neurons = inputs;

		weights = (T) CPUMatrix.randomMatrix(neurons, inputs, new Random(), -1, 1);
	}

	@Override
	public void setWeights(T newWeights) {
		weights = newWeights;
	}

}
