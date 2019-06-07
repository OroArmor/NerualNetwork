package oroarmor.training.models;

import java.util.ArrayList;

import oroarmor.layer.Layer;
import oroarmor.matrix.Matrix;

public class TotalError extends TrainingModel {

	double trainingRate;

	public TotalError(double trainingRate) {
		super(trainingRate);
		this.trainingRate = trainingRate;
	}

	public void fixErrors(ArrayList<Layer> layers, Matrix[] layerOutputs, Matrix expectedOutput, Matrix inputs) {

//		Matrix outputErrors = expectedOutput.subtractMatrix(layerOutputs[layerOutputs.length - 1]).pow(2).divide(2);

		Matrix[] deltas = new Matrix[layers.size()];

		int layerIndex = deltas.length - 1;
		Matrix currentOutput = layerOutputs[layerIndex];
		deltas[layerIndex] = (currentOutput.subtractMatrix(expectedOutput))
				.hadamard((layerOutputs[layerIndex]
						.getDerivative(layers.get(layerIndex).getMatrixFunction())));

		layerIndex--;

		for (; layerIndex >= 0; layerIndex--) {
			deltas[layerIndex] = (// ---UPPER LAYER---
			layers.get(layerIndex + 1).getWeights().transpose().multiplyMatrix(// get the weights from i+1 and transpose
																				// them.
					deltas[layerIndex + 1] // multiply those transposed weights with the deltas from the next layer
			)).hadamard( // get the hadamard product of UPPER LAYER and CURRENT LAYER
					// ---CURRENT LAYER---
					(layerOutputs[layerIndex]).getDerivative(layers.get(layerIndex).getMatrixFunction()))// get the anti-derivative of the
																				// current weights and past output
			;
		}

		for (int i = 0; i < layers.size(); i++) {
			Matrix delEoverDelWeight = deltas[i]
					.multiplyMatrix((i > 0) ? layerOutputs[i - 1].transpose() : inputs.transpose());
			Layer currentLayer = layers.get(i);
			currentLayer.setWeights(currentLayer.getWeights()
					.subtractMatrix(delEoverDelWeight.multiply(trainingRate)
					));
		}
	}
}