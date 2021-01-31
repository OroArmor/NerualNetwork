package com.oroarmor.neural_network.lib.training;

import com.oroarmor.neural_network.lib.matrix.Matrix;
import com.oroarmor.neural_network.lib.network.NeuralNetwork;
import com.oroarmor.neural_network.lib.training.models.TrainingModel;

public class Trainer<T extends Matrix<T>> implements Runnable {

	GetData<T> getInput;
	GetData<T> getOutput;
	NeuralNetwork<T> network;
	TrainingModel model;

	public Trainer(GetData<T> getInput, GetData<T> getOutput, NeuralNetwork<T> network, TrainingModel model) {
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

	public GetData<T> getGetInput() {
		return getInput;
	}

	public void setGetInput(GetData<T> getInput) {
		this.getInput = getInput;
	}

	public GetData<T> getGetOutput() {
		return getOutput;
	}

	public void setGetOutput(GetData<T> getOutput) {
		this.getOutput = getOutput;
	}

	public NeuralNetwork<T> getNetwork() {
		return network;
	}

	public void setNetwork(NeuralNetwork<T> network) {
		this.network = network;
	}

	@Override
	public void run() {
		for (int i = 0; i < Integer.parseInt(getInput.globalArgs[1]); i++) {
			T input = getInput.getData(new String[] { i + "" });
			T output = getOutput.getData(new String[] { i + "" });
			network.train(input, output, model);
		}
	}

}
