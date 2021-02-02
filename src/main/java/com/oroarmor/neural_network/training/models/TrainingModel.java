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
