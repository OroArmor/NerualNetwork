package com.oroarmor.neural_network.network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.stream.Collectors;

import com.oroarmor.neural_network.layer.Layer;
import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.training.models.TrainingModel;
import com.oroarmor.neural_network.util.SerializationIndexer;

/**
 * A neural network class
 * @param <T> The matrix type
 * @author OroArmor
 */
public class NeuralNetwork<T extends Matrix<T>> extends AbstractNetwork<T> implements Serializable {
    private static final long serialVersionUID = SerializationIndexer.NEURAL_NETWORK_ID;

    /**
     * Creates a new network with the given inputs
     * @param inputNeurons The number of inputs
     */
    public NeuralNetwork(int inputNeurons) {
        super(inputNeurons, new ArrayList<>());
    }

    public T feedForward(T inputs) {
        for (Layer<T> layer : layers) {
            inputs = layer.feedForward(inputs);
        }
        return inputs;
    }


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

    public NeuralNetwork<CPUMatrix> convertAllToCPU() {
        NeuralNetwork<CPUMatrix> newNetwork = new NeuralNetwork<>(inputs);
        newNetwork.trains = this.trains;
        newNetwork.layers = layers.stream().map(Layer::convertToCPU)
                .collect(Collectors.toList());

        return newNetwork;
    }
}
