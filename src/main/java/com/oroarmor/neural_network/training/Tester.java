package com.oroarmor.neural_network.training;

import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.network.NeuralNetwork;

/**
 * Tests the network with testing data.
 * A test is considered "correct" when the max output is also the max correct output
 * @param <T> The matrix class
 * @author OroArmor
 */
public class Tester<T extends Matrix<T>> implements Runnable {
    /**
     * The number of correct trains
     */
    public static transient int numCorrect = 0;

    /**
     * The input provider
     */
    protected DataProvider<T> getInput;

    /**
     * The output provider
     */
    protected DataProvider<T> getOutput;

    /**
     * The network to test
     */
    protected NeuralNetwork<T> network;

    /**
     * The threshold to consider when a value is considered satisfactory
     */
    protected double threshold;

    /**
     * Creates a new {@link Tester}
     * @param getInput The input provider
     * @param getOutput The output provider
     * @param network The network
     */
    public Tester(DataProvider<T> getInput, DataProvider<T> getOutput, NeuralNetwork<T> network, double threshold) {
        this.getInput = getInput;
        this.getOutput = getOutput;
        this.network = network;
        this.threshold = threshold;
    }

    @Override
    public void run() {
        for (int i = 0; i < (Integer) getInput.globalArgs[1]; i++) {
            T input = getInput.getData(new Object[]{i});
            T output = getOutput.getData(new Object[]{i});
            T calculatedOutput = network.feedForward(input);
            int maxIndex = calculatedOutput.getMaxIndex();
            if (maxIndex == output.getMaxIndex() && calculatedOutput.getValue(maxIndex, 0) > threshold) {
                addCorrect();
            }
        }
    }

    public synchronized void addCorrect() {
        numCorrect++;
    }
}
