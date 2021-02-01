package com.oroarmor.neural_network.network;

import java.io.Serializable;
import java.util.List;

import com.oroarmor.neural_network.layer.Layer;
import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.training.models.TrainingModel;


/**
 * An abstract implementation for Neural Networks
 * @param <T> The matrix class
 * @author OroArmor
 */
public abstract class AbstractNetwork<T extends Matrix<T>> implements Serializable {
	/**
	 * The layers for the network
	 */
	protected List<Layer<T>> layers;

	/**
	 * The number of inputs for the network
	 */
	protected int inputs;

	/**
	 * The number of training attempts on the network
	 */
	protected int trains;

	protected AbstractNetwork(int inputs, List<Layer<T>> layers) {
		this.inputs = inputs;
		this.layers = layers;
	}

	/**
	 * Adds a layer to the network
	 * @param layer The new layer
	 */
	public void addLayer(Layer<T> layer) {
		if (layers.isEmpty()) {
			layer.setup(inputs);
		} else {
			layer.setup(layers.get(layers.size() - 1).getOutputNeurons());
		}
		layers.add(layer);
	}

	/**
	 * Feeds the inputs through all layers
	 * @param inputs The inputs
	 * @return The output
	 */
	public abstract T feedForward(T inputs);

	/**
	 *
	 * @param layerIndex The index
	 * @return The layer at the index
	 */
	public Layer<T> getLayer(int layerIndex) {
		return layers.get(layerIndex);
	}

	/**
	 *
	 * @return The number of training attempts
	 */
	public int getTrainingAttempts() {
		return trains;
	}

	/**
	 * Trains the network once
	 * @param input The input matrix
	 * @param output The output matrix
	 * @param model The training model
	 */
	public abstract void train(T input, T output, TrainingModel model);

	/**
	 * Converts the neural network to completely CPU based
	 * @return A CPU neural network
	 */
	public abstract AbstractNetwork<CPUMatrix> convertAllToCPU();
}
