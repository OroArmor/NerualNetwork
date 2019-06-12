package oroarmor.neuralnetwork.training;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.network.NeuralNetwork;
import oroarmor.neuralnetwork.training.models.TrainingModel;

public class Trainer implements Runnable {

	GetData getInput;
	GetData getOutput;
	NeuralNetwork network;
	TrainingModel model;

	public Trainer(GetData getInput, GetData getOutput, NeuralNetwork network, TrainingModel model) {
		this.getInput = getInput;
		this.getOutput = getOutput;
		this.network = network;
		this.model = model;
	}

	public TrainingModel getModel() {
		return model;
	}

	public void setModel(TrainingModel model) {
		this.model = model;
	}

	public GetData getGetInput() {
		return getInput;
	}

	public void setGetInput(GetData getInput) {
		this.getInput = getInput;
	}

	public GetData getGetOutput() {
		return getOutput;
	}

	public void setGetOutput(GetData getOutput) {
		this.getOutput = getOutput;
	}

	public NeuralNetwork getNetwork() {
		return network;
	}

	public void setNetwork(NeuralNetwork network) {
		this.network = network;
	}

	@Override
	public void run() {
		for (int i = 0; i < Integer.parseInt(getInput.globalArgs[1]); i++) {
			Matrix input = getInput.getData(new String[] { i + "" });
			Matrix output = getOutput.getData(new String[] { i + "" });
			network.train(input, output, model);
		}
	}

}
