/*
 * MIT License
 *
 * Copyright (c) 2021 OroArmor
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.oroarmor.neural_network.training.models;

import java.util.ArrayList;
import java.util.List;

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
    public synchronized <T extends Matrix<T>> void fixErrors(List<Layer<T>> layers, T[] layerOutputs,
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