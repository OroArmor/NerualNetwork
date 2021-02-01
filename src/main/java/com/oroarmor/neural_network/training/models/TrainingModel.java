package com.oroarmor.neural_network.training.models;

import java.util.ArrayList;
import java.util.List;

import com.oroarmor.neural_network.layer.Layer;
import com.oroarmor.neural_network.matrix.Matrix;

/**
 * An abstract training model
 * @author OroArmor
 */
public abstract class TrainingModel {
    /**
     * The training rate for the model
     */
    protected double trainingRate;

    public TrainingModel(double trainingRate) {
        this.trainingRate = trainingRate;
    }

    /**
     * Fixes the errors for one training step
     * @param layers A list of the layers
     * @param layerOutputs The outputs for the layers
     * @param expected The expected output
     * @param input The real output
     * @param <T> The matrix class
     */
    public abstract <T extends Matrix<T>> void fixErrors(List<Layer<T>> layers, T[] layerOutputs, T expected, T input);
}
