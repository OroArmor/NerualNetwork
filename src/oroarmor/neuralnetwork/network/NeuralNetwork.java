package oroarmor.neuralnetwork.network;

import java.io.Serializable;
import java.util.ArrayList;

import oroarmor.neuralnetwork.layer.Layer;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.training.models.TrainingModel;

public class NeuralNetwork implements Serializable {
	private static final long serialVersionUID = 1L;

	ArrayList<Layer> layers;
	int inputs;
	int trains;

	public NeuralNetwork(int inputNeurons) {
		inputs = inputNeurons;
		layers = new ArrayList<Layer>();
	}

	public void addLayer(Layer layer) {
		if (layers.isEmpty()) {
			layer.setup(inputs);
		} else {
			layer.setup(layers.get(layers.size() - 1).getOutputNeurons());
		}
		layers.add(layer);
	}

	public Matrix feedFoward(Matrix inputs) {
		for (Layer layer : layers) {
			inputs = layer.feedFoward(inputs);
		}
		return inputs;
	}

	public Layer getLayer(int layerIndex) {
		return layers.get(layerIndex);
	}

	public int getTrainingAttemps() {
		return this.trains;
	}

	public synchronized void train(Matrix input, Matrix output, TrainingModel model) {
		trains++;
		Matrix[] layerOutputs = new Matrix[layers.size()];
		int i = 0;
		for (Layer layer : layers) {
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
