package com.oroarmor.neural_network.network;

import com.oroarmor.neural_network.matrix.Matrix;

/**
 * An {@link AutoEncoder} network tries to replicate the training data that is given to it.
 * @param <T> The Matrix class
 * @author OroArmor
 */
public class AutoEncoder<T extends Matrix<T>> extends NeuralNetwork<T> {
	private static final long serialVersionUID = 2L;

	/**
	 * The layer that the encoder uses as an input
	 */
	protected int encoderLayer;

	/**
	 * Creates a new {@link AutoEncoder}
	 * @param inputNeurons The number of input neurons
	 * @param encoderLayer The layer for the encoder input
	 */
	public AutoEncoder(int inputNeurons, int encoderLayer) {
		super(inputNeurons);
		this.encoderLayer = encoderLayer;
	}

	@Override
	public T feedForward(T inputs) {
		if (layers.size() < encoderLayer) {
			return null;
		}

		for (int i = encoderLayer + 1; i < layers.size(); i++) {
			inputs = layers.get(i).feedForward(inputs);
		}
		return inputs;
	}
}
