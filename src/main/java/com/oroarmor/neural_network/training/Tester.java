package com.oroarmor.neural_network.training;

import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.network.NeuralNetwork;

public class Tester<T extends Matrix<T>> implements Runnable {
    public static transient int numCorrect = 0;
    protected GetData<T> getInput;
    protected GetData<T> getOutput;
    protected NeuralNetwork<T> network;

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
            T input = getInput.getData(new String[]{i + ""});
            T output = getOutput.getData(new String[]{i + ""});
            if (network.feedFoward(input).getMax() == output.getMax()) {
                addCorrect();
            }
        }
    }

    public synchronized void addCorrect() {
        numCorrect++;
    }
}
