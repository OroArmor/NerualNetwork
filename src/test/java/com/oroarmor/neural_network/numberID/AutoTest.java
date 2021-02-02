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

package com.oroarmor.neural_network.numberID;

import java.io.IOException;
import java.util.Objects;
import java.util.Random;

import com.oroarmor.neural_network.layer.FeedForwardLayer;
import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix;
import com.oroarmor.neural_network.matrix.function.SoftMaxFunction;
import com.oroarmor.neural_network.network.AutoEncoder;
import com.oroarmor.neural_network.network.NetworkSaver;
import com.oroarmor.neural_network.training.DataProvider;
import com.oroarmor.neural_network.training.Trainer;
import com.oroarmor.neural_network.training.models.TotalError;
import processing.core.PApplet;

public class AutoTest extends PApplet {
    AutoEncoder<CPUMatrix> encoder;
    boolean reset = true;

    public static int numImages = 12000;
    public static CPUMatrix[] images = new CPUMatrix[numImages];

    public static void main(String[] args) {
        System.out.print("Loading Data");
        long startTime = System.currentTimeMillis();
        byte[] dataBytes = new byte[0];

        try {
            dataBytes = Objects.requireNonNull(AutoTest.class.getClassLoader().getResourceAsStream("numberID/train/train-images-idx3-ubyte-uncompressed")).readAllBytes();
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int k = 0; k < numImages; k++) {
            double[] imageDoubles = new double[28 * 28];
            for (byte i = 0; i < 28; i++) {
                for (byte j = 0; j < 28; j++) {
                    byte value = dataBytes[16 + j * 28 + i + k * 28 * 28];
                    imageDoubles[i * 28 + j] = (value >= 0 ? value : value + 255d) / 255d;
                }
            }
            images[k] = new CPUMatrix(imageDoubles, 28 * 28, 1);
            if (k % 1200 == 0) {
                System.out.print(".");
            }
        }
        System.out.println("\nData loaded in: " + (System.currentTimeMillis() - startTime) / 1000 + " seconds");
        PApplet.main("com.oroarmor.neural_network.numberID.AutoTest");
    }

    @Override
    public void setup() {
        encoder = NetworkSaver.loadNetworkFromFile("C:\\oroarmor\\", "test.nn", AutoEncoder.class);
        if (encoder == null || reset) {
            encoder = new AutoEncoder<>(28 * 28, 2);
            encoder.addLayer(new FeedForwardLayer<>(128, Matrix.MatrixType.CPU));
            encoder.addLayer(new FeedForwardLayer<>(64, Matrix.MatrixType.CPU));
            encoder.addLayer(new FeedForwardLayer<>(32, Matrix.MatrixType.CPU));
            encoder.addLayer(new FeedForwardLayer<>(64, Matrix.MatrixType.CPU));
            encoder.addLayer(new FeedForwardLayer<>(128, Matrix.MatrixType.CPU));
            encoder.addLayer(new FeedForwardLayer<>(28 * 28, Matrix.MatrixType.CPU));
        }

        int numThreads = 12;
        int numRepeats = 1;

        Thread[] threads = new Thread[numThreads];

        for (int repeat = 0; repeat < numRepeats; repeat++) {
            long millis = System.currentTimeMillis();
            for (int thread = 0; thread < numThreads; thread++) {
                DataProvider<CPUMatrix> getData = new DataProvider<>(new Object[]{thread, numImages / numThreads}) {
                    @Override
                    public CPUMatrix getData(Object[] args) {
                        int image = (Integer) args[0];
                        Integer imagesPerThread = (Integer) globalArgs[1];
                        Integer threads = (Integer) globalArgs[0];
                        int threadIndex = imagesPerThread * threads;
                        return images[image + threadIndex];
                    }
                };
                Trainer<CPUMatrix> trainer = new Trainer<>(getData, getData, encoder, new TotalError(10));
                threads[thread] = new Thread(trainer);
            }

            for (Thread thread : threads) {
                thread.start();
            }

            for (Thread thread : threads) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            System.out.println(repeat + ": " + (System.currentTimeMillis() - millis));
            NetworkSaver.saveNetworkToFile(encoder, "C:\\oroarmor\\", "test.nn");
        }
        noStroke();
    }

    @Override
    public void settings() {
        size(280, 280);
    }

    @Override
    public void draw() {
        frameRate(1);
        background(0, 0, 0);

        encoder.train(images[frameCount % numImages], images[frameCount % numImages], new TotalError(0.1));

        CPUMatrix test = Matrix.randomMatrix(Matrix.MatrixType.CPU, 32, 1, new Random(), -1, 1);
        assert test != null;
//        test.applyFunction(new SoftMaxFunction());

        double[] values = encoder.feedForward(test).getValues();
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                fill((int) (values[i * 28 + j] * 255d));
                rect(i * 10, j * 10, 10, 10);
            }
        }
        System.out.println("New Frame");
    }
}
