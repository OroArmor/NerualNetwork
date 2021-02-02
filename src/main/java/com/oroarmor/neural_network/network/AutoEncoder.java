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

import java.util.stream.Collectors;

import com.oroarmor.neural_network.layer.Layer;
import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.util.SerializationIndexer;

/**
 * An {@link AutoEncoder} network tries to replicate the training data that is given to it.
 * @param <T> The Matrix class
 * @author OroArmor
 */
public class AutoEncoder<T extends Matrix<T>> extends NeuralNetwork<T> {
	private static final long serialVersionUID = SerializationIndexer.AUTO_ENCODER_ID;

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

	public AutoEncoder<CPUMatrix> convertAllToCPU() {
		AutoEncoder<CPUMatrix> newNetwork = new AutoEncoder<>(inputs, encoderLayer);
		newNetwork.trains = this.trains;
		newNetwork.layers = layers.stream().map(Layer::convertToCPU)
				.collect(Collectors.toList());

		return newNetwork;
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
