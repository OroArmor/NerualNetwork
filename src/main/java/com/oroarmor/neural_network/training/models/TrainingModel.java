package com.oroarmor.neural_network.training.models;

import java.util.ArrayList;

import com.oroarmor.neural_network.layer.Layer;
import com.oroarmor.neural_network.matrix.Matrix;

public abstract class TrainingModel {
    protected double trainingRate;

    public TrainingModel(double trainingRate) {
        this.trainingRate = trainingRate;
    }

    public abstract <T extends Matrix<T>> void fixErrors(ArrayList<Layer<T>> layers, T[] layerOutputs, T expected,
                                                         T input);
}
