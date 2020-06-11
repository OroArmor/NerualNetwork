package oroarmor.neuralnetwork.training.models;

import java.util.ArrayList;

import oroarmor.neuralnetwork.layer.Layer;
import oroarmor.neuralnetwork.matrix.Matrix;

public abstract class TrainingModel {

	protected double trainingRate;

	public TrainingModel(double trainingRate) {
		this.trainingRate = trainingRate;
	}

	public abstract void fixErrors(ArrayList<Layer> layers, Matrix[] layerOutputs, Matrix expected, Matrix input);

}
