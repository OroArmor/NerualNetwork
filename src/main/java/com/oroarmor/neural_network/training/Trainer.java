package com.oroarmor.neural_network.training;

import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.network.NeuralNetwork;
import com.oroarmor.neural_network.training.models.TrainingModel;

/**
 * A trainer for neural networks
 *
 * @param <T> The matrix class
 * @author OroArmor
 */
public class Trainer<T extends Matrix<T>> implements Runnable {
    /**
     * The input provider
     */
    protected DataProvider<T> getInput;

    /**
     * The output provider
     */
    protected DataProvider<T> getOutput;

    /**
     * The network
     */
    protected NeuralNetwork<T> network;

    /**
     * The training model
     */
    protected TrainingModel model;

    /**
     * Creates a new {@link Trainer}
     *
     * @param getInput  The input provider
     * @param getOutput The output provider
     * @param network   The network
     * @param model     The training model
     */
    public Trainer(DataProvider<T> getInput, DataProvider<T> getOutput, NeuralNetwork<T> network, TrainingModel model) {
        this.getInput = getInput;
        this.getOutput = getOutput;
        this.network = network;
        this.model = model;
    }

    @Override
    public void run() {
        for (int i = 0; i < (Integer) getInput.globalArgs[1]; i++) {
            T input = getInput.getData(new Object[]{i});
            T output = getOutput.getData(new Object[]{i});
            network.train(input, output, model);
        }
    }
}
