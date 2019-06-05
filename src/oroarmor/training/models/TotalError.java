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

	public void fixErrors(ArrayList<Layer> layers, Matrix[] layerOutputs, Matrix expectedOutput) {

//		Matrix outputErrors = expectedOutput.subtractMatrix(layerOutputs[layerOutputs.length - 1]).pow(2).divide(2);

		Matrix[] deltas = new Matrix[layers.size()];

		int layerIndex = deltas.length - 1;
		Matrix currentOutput = layerOutputs[layerIndex];
		deltas[layerIndex] = (currentOutput.subtractMatrix(expectedOutput))
				.multiplyMatrix(((layers.get(layerIndex).getWeights().multiplyMatrix(currentOutput))
						.applyFunction(layers.get(layerIndex).getMatrixFunction())).transpose());

		deltas[layerIndex].print();

		layerIndex--;

		for (; layerIndex >= 0; layerIndex--) {
			System.out.println(layerIndex);
			deltas[layerIndex] = (layers.get(layerIndex+1).getWeights().transpose().multiplyMatrix(deltas[layerIndex+1]))
					.multiplyMatrix(
							
							);
		}

		for (int i = layers.size() - 1; i >= 0; i--) {
		}
	}
}