package com.oroarmor.neural_network.lib.training.models;

import java.util.ArrayList;

import com.oroarmor.neural_network.lib.layer.Layer;
import com.oroarmor.neural_network.lib.matrix.Matrix;

public abstract class TrainingModel {

	protected double trainingRate;

	public TrainingModel(double trainingRate) {
		this.trainingRate = trainingRate;
	}

	public abstract <T extends Matrix<T>> void fixErrors(ArrayList<Layer<T>> layers, T[] layerOutputs, T expected,
			T input);

}
