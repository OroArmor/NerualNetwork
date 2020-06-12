package oroarmor.neuralnetwork.training.models;

import java.util.ArrayList;

import oroarmor.neuralnetwork.layer.Layer;
import oroarmor.neuralnetwork.matrix.Matrix;

public abstract class TrainingModel {

	protected double trainingRate;

	public TrainingModel(double trainingRate) {
		this.trainingRate = trainingRate;
	}

	public abstract <T extends Matrix<T>> void fixErrors(ArrayList<Layer<T>> layers, T[] layerOutputs, T expected,
			T input);

}
