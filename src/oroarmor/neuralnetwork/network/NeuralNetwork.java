package oroarmor.neuralnetwork.network;

import java.io.Serializable;
import java.util.ArrayList;

import oroarmor.neuralnetwork.layer.Layer;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.training.models.TrainingModel;

public class NeuralNetwork<T extends Matrix<T>> implements Serializable {
	private static final long serialVersionUID = 1L;

	ArrayList<Layer<T>> layers;
	int inputs;
	int trains;

	public NeuralNetwork(int inputNeurons) {
		inputs = inputNeurons;
		layers = new ArrayList<>();
	}

	public void addLayer(Layer<T> layer) {
		if (layers.isEmpty()) {
			layer.setup(inputs);
		} else {
			layer.setup(layers.get(layers.size() - 1).getOutputNeurons());
		}
		layers.add(layer);
	}

	public T feedFoward(T inputs) {
		for (Layer<T> layer : layers) {
			inputs = layer.feedFoward(inputs);
		}
		return inputs;
	}

	public Layer<T> getLayer(int layerIndex) {
		return layers.get(layerIndex);
	}

	public int getTrainingAttemps() {
		return trains;
	}

	@SuppressWarnings("unchecked")
	public synchronized void train(T input, T output, TrainingModel model) {
		trains++;

		T[] layerOutputs = (T[]) new Matrix[layers.size()];
		int i = 0;
		for (Layer<T> layer : layers) {
			if (i == 0) {
				layerOutputs[i] = layer.feedFoward(input);
			} else {
				layerOutputs[i] = layer.feedFoward(layerOutputs[i - 1]);
			}

			i++;
		}

		model.fixErrors(layers, layerOutputs, output, input);
	}
}
