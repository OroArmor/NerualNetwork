package oroarmor.neuralnetwork.training;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.network.NeuralNetwork;

public class Tester<T extends Matrix<T>> implements Runnable {

	GetData<T> getInput;
	GetData<T> getOutput;
	NeuralNetwork<T> network;
	public static int numCorrect = 0;

	public Tester(GetData<T> getInput, GetData<T> getOutput, NeuralNetwork<T> network) {
		this.getInput = getInput;
		this.getOutput = getOutput;
		this.network = network;

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
			if (network.feedFoward(input).getMax() == output.getMax()) {
				addCorrect();
			}
		}
	}

	public synchronized void addCorrect() {
		numCorrect++;
	}

}
