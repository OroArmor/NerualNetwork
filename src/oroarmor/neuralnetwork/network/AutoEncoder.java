package oroarmor.neuralnetwork.network;

import oroarmor.neuralnetwork.matrix.Matrix;

public class AutoEncoder extends NeuralNetwork {

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;

	int encoderLayer = 0;

	public AutoEncoder(int inputNeurons, int encoderLayer) {
		super(inputNeurons);
		this.encoderLayer = encoderLayer;
	}

	@Override
	public Matrix feedFoward(Matrix inputs) {
		if (layers.size() < encoderLayer) {
			return null;
		}

		for (int i = encoderLayer + 1; i < layers.size(); i++) {
			inputs = layers.get(i).feedFoward(inputs);
		}
		return inputs;
	}

}
