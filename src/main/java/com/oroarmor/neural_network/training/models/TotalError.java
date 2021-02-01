package com.oroarmor.neural_network.training.models;

import java.util.ArrayList;

import com.oroarmor.neural_network.layer.Layer;
import com.oroarmor.neural_network.matrix.Matrix;

/**
 * A {@link TrainingModel} using the total error from training attempts
 * @author OroArmor
 */
public class TotalError extends TrainingModel {
    /**
     * Creates a new {@link TotalError}
     * @param trainingRate The training rate for the model
     */
    public TotalError(double trainingRate) {
        super(trainingRate);
    }

    @SuppressWarnings("unchecked")
    @Override
    public synchronized <T extends Matrix<T>> void fixErrors(ArrayList<Layer<T>> layers, T[] layerOutputs,
                                                             T expectedOutput, T inputs) {
        T outputErrors = expectedOutput.subtractMatrix(layerOutputs[layerOutputs.length - 1]);

        double totalError = Math.pow(outputErrors.sum(), 2);

        T[] deltas = (T[]) new Matrix[layers.size()];

        int layerIndex = deltas.length - 1;
        T currentOutput = layerOutputs[layerIndex];
        deltas[layerIndex] = currentOutput.subtractMatrix(expectedOutput)
                .hadamard(layerOutputs[layerIndex].getDerivative(layers.get(layerIndex).getMatrixFunction()));

        layerIndex--;

        for (; layerIndex >= 0; layerIndex--) {
            deltas[layerIndex] = layers.get(layerIndex + 1).getWeights().transpose().multiplyMatrix(// get the weights from i+1 and transpose them.
                    deltas[layerIndex + 1] // multiply those transposed weights with the deltas from the next layer
            ).hadamard( // get the hadamard product of UPPER LAYER and CURRENT LAYER
                    // ---CURRENT LAYER---
                    layerOutputs[layerIndex].getDerivative(layers.get(layerIndex).getMatrixFunction()))// get the anti-derivative of the current weights and past output
            ;
        }

        for (int i = 0; i < layers.size(); i++) {
            T delEoverDelWeight = deltas[i]
                    .multiplyMatrix(i > 0 ? layerOutputs[i - 1].transpose() : inputs.transpose());
            Layer<T> currentLayer = layers.get(i);
            currentLayer.setWeights(currentLayer.getWeights()
                    .subtractMatrix(delEoverDelWeight.multiply(trainingRate * totalError * 10)));
        }
    }
}