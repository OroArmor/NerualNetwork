package oroarmor.training.models;

import java.util.ArrayList;

import oroarmor.layer.Layer;
import oroarmor.matrix.Matrix;

public abstract class TrainingModel {

	public TrainingModel(double trainingRate) {
		// TODO Auto-generated constructor stub
	}

	public abstract void fixErrors(ArrayList<Layer> layers, Matrix[] layerOutputs, Matrix expected);

}
