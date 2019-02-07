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

		Matrix outputErrors = expectedOutput.subtractMatrix(layerOutputs[layerOutputs.length - 1]);

		for (int i = layers.size() - 1; i >= 0; i--) {

			Layer currentLayer = layers.get(i);

			Matrix gradient = layerOutputs[i].getDerivative(currentLayer.getMatrixFunction());

			gradient = gradient.multiplyMatrix(outputErrors.transpose());
			gradient = gradient.multiply(trainingRate);

			Matrix weightDelta = gradient.multiplyMatrix(currentLayer.getWeights());

			currentLayer.setWeights(currentLayer.getWeights().addMatrix(weightDelta));

			currentLayer.setBias(currentLayer.getBias().addMatrix(gradient.collapseRows().divide(2)));
		}
	}
}