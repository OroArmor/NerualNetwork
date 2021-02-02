/*
 * MIT License
 *
 * Copyright (c) 2021 OroArmor
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.oroarmor.neural_network.network;

import java.io.Serializable;
import java.util.List;

import com.oroarmor.neural_network.layer.Layer;
import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.training.models.TrainingModel;
import com.oroarmor.neural_network.util.SerializationIndexer;


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
