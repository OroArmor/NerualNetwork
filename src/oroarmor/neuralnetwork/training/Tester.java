package oroarmor.neuralnetwork.training;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.network.NeuralNetwork;

public class Tester implements Runnable {

	GetData getInput;
	GetData getOutput;
	NeuralNetwork network;
	public static int numCorrect = 0;

	public Tester(GetData getInput, GetData getOutput, NeuralNetwork network) {
		this.getInput = getInput;
		this.getOutput = getOutput;
		this.network = network;

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
			if (network.feedFoward(input).getMax() == output.getMax()) {
				addCorrect();
			}
		}
	}

	public synchronized void addCorrect() {
		numCorrect++;
	}

}
