package oroarmor.neuralnetwork.network;

import java.io.Serializable;
import java.util.ArrayList;

import oroarmor.neuralnetwork.layer.Layer;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.training.models.TrainingModel;

public abstract class ANetwork implements Serializable {

	private static final long serialVersionUID = 1L;

	public int inputs;
	public int trains;
	public ArrayList<Layer> layers;
	
	public abstract Matrix feedFoward(Matrix inputs);
	public abstract Layer getLayer(int index);
	public abstract int getTrainingAttemps();
	public abstract void train(Matrix input, Matrix output, TrainingModel model);
}
