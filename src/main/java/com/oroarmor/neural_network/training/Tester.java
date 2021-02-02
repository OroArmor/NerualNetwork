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

package com.oroarmor.neural_network.training;

import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.SoftMaxFunction;
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
            if (maxIndex == output.getMaxIndex() && calculatedOutput.applyFunction(new SoftMaxFunction()).getValue(maxIndex, 0) > threshold) {
                addCorrect();
            }
        }
    }

    public synchronized void addCorrect() {
        numCorrect++;
    }
}
