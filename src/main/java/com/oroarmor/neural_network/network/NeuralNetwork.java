package com.oroarmor.neural_network.network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.stream.Collectors;

import com.oroarmor.neural_network.layer.Layer;
import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.training.models.TrainingModel;

/**
 * A neural network class
 * @param <T> The matrix type
 * @author OroArmor
 */
public class NeuralNetwork<T extends Matrix<T>> implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * The layers for the network
     */
    protected ArrayList<Layer<T>> layers;

    /**
     * The number of inputs for the network
     */
    protected int inputs;

    /**
     * The number of training attempts on the network
     */
    protected int trains;

    /**
     * Creates a new network with the given inputs
     * @param inputNeurons The number of inputs
     */
    public NeuralNetwork(int inputNeurons) {
        inputs = inputNeurons;
        layers = new ArrayList<>();
    }

    /**
     * Adds a layer to the network
     * @param layer The new layer
     */
    public void addLayer(Layer<T> layer) {
        if (layers.isEmpty()) {
            layer.setup(inputs);
        } else {
            layer.setup(layers.get(layers.size() - 1).getOutputNeurons());
        }
        layers.add(layer);
    }

    /**
     * Feeds the inputs through all layers
     * @param inputs The inputs
     * @return The output
     */
    public T feedForward(T inputs) {
        for (Layer<T> layer : layers) {
            inputs = layer.feedForward(inputs);
        }
        return inputs;
    }

    /**
     *
     * @param layerIndex The index
     * @return The layer at the index
     */
    public Layer<T> getLayer(int layerIndex) {
        return layers.get(layerIndex);
    }

    /**
     *
     * @return The number of training attempts
     */
    public int getTrainingAttempts() {
        return trains;
    }

    /**
     * Trains the network once
     * @param input The input matrix
     * @param output The output matrix
     * @param model The training model
     */
    @SuppressWarnings("unchecked")
    public synchronized void train(T input, T output, TrainingModel model) {
        trains++;

        T[] layerOutputs = (T[]) new Matrix[layers.size()];
        int i = 0;
        for (Layer<T> layer : layers) {
            if (i == 0) {
                layerOutputs[i] = layer.feedForward(input);
            } else {
                layerOutputs[i] = layer.feedForward(layerOutputs[i - 1]);
            }

            i++;
        }

        model.fixErrors(layers, layerOutputs, output, input);
    }

    /**
     * Converts the neural network to completely CPU based
     * @return A CPU neural network
     */
    public NeuralNetwork<CPUMatrix> convertAllToCPU() {
        NeuralNetwork<CPUMatrix> newNetwork = new NeuralNetwork<>(inputs);
        newNetwork.trains = this.trains;
        newNetwork.layers = (ArrayList<Layer<CPUMatrix>>) layers.stream().map(Layer::convertToCPU)
                .collect(Collectors.toList());

        return newNetwork;
    }
}
