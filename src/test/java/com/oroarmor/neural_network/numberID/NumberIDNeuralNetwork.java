package com.oroarmor.neural_network.numberID;

import java.io.File;
import java.io.FileInputStream;
import java.text.DecimalFormat;

import com.oroarmor.neural_network.layer.FeedForwardLayer;
import com.oroarmor.neural_network.layer.SoftMaxLayer;
import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix.MatrixType;
import com.oroarmor.neural_network.matrix.function.SoftMaxFunction;
import com.oroarmor.neural_network.network.NetworkSaver;
import com.oroarmor.neural_network.network.NeuralNetwork;
import com.oroarmor.neural_network.training.DataProvider;
import com.oroarmor.neural_network.training.Tester;
import com.oroarmor.neural_network.training.Trainer;
import com.oroarmor.neural_network.training.models.TotalError;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;

public class NumberIDNeuralNetwork extends PApplet {
    boolean reset = false;
    NeuralNetwork<CPUMatrix> numberIDNetwork;

    public static void main(String[] args) {
        PApplet.main("com.oroarmor.neural_network.numberID.NumberIDNeuralNetwork");
    }

    public CPUMatrix getImageData(String string) {
        return getImageData(loadImage(string));
    }

    public CPUMatrix getImageData(PImage image) {
        image.loadPixels();
        double[] CPUMatrixArray = new double[image.height * image.width];
        for (int i = 0; i < CPUMatrixArray.length; i++) {
            CPUMatrixArray[i] = (double) brightness(image.pixels[i]) / 255f;
        }
        return new CPUMatrix(CPUMatrixArray, CPUMatrixArray.length, 1);
    }

    @Override
    public void settings() {
        size(280 * 2, 400);
    }

    @Override
    public void setup() {
        background(0);
        noStroke();
        numberIDNetwork = NetworkSaver.loadNetworkFromFile("C:\\oroarmor\\numberID\\", "numberIDNetwork.nn", NeuralNetwork.class);

        if (numberIDNetwork == null || reset) {
            numberIDNetwork = new NeuralNetwork<>(28 * 28);
            numberIDNetwork.addLayer(new FeedForwardLayer<>(64, MatrixType.CPU));
            numberIDNetwork.addLayer(new FeedForwardLayer<>(32, MatrixType.CPU));
            numberIDNetwork.addLayer(new FeedForwardLayer<>(16, MatrixType.CPU));
            numberIDNetwork.addLayer(new FeedForwardLayer<>(16, MatrixType.CPU));
            numberIDNetwork.addLayer(new FeedForwardLayer<>(10, MatrixType.CPU));
        }
        test();
        train();
    }

    @Override
    public void draw() {
        if (mousePressed && mouseX < 280) {
            stroke(255, 255 / 2f);
            for (int i = 0; i < 5; i++) {
                strokeWeight(25 - i * 5);
                line(mouseX, mouseY, pmouseX, pmouseY);
            }
        }
        fill(255);

        noStroke();
        rect(0, 280, 280 * 2, 120);

        PImage myTest = new PImage(280, 280);
        myTest.loadPixels();

        myTest.blend(g.copy(), 0, 0, 280, 280, 0, 0, 280, 280, PConstants.ADD);

        myTest.resize(28, 28);
        myTest.updatePixels();

        image(myTest, 280, 0, 280, 280);

        stroke(255);
        strokeWeight(1);
        line(280, 0, 280, 280);
        noStroke();

        CPUMatrix outputs = numberIDNetwork.feedForward(getImageData(myTest)).applyFunction(new SoftMaxFunction());
        noStroke();
        for (int i = 0; i < 10; i++) {
            fill((float) (1 - outputs.getValue(i, 0)) * 255f, (float) outputs.getValue(i, 0) * 255f, 0);
            rect(i * 2 * 280 / 10f, 280, 28 * 2, 120);

            fill(0);
            text(i + ": \n" + outputs.getValue(i, 0), i * 28 * 2, 300);
        }
    }

    @Override
    public void keyPressed() {
        if (key == 'c') {
            background(0);
        }
    }

    public void train() {
        long start = System.currentTimeMillis();
        for (int repeats = 0; repeats < 16; repeats++) {
            int threads = 12;
            Thread[] trainingThreads = new Thread[threads];

            int numImages = 60000;

            for (int i = 0; i < threads; i++) {
                DataProvider<CPUMatrix> getInputs = new DataProvider<>(new Object[]{i, numImages / threads}) {
                    @Override
                    public CPUMatrix getData(Object[] args) {
                        return getImageData(loadImage("C:\\oroarmor\\numberID\\train\\images\\" + ((Integer) globalArgs[0] * (Integer) globalArgs[1] + (Integer) args[0]) + ".png"));
                    }
                };

                DataProvider<CPUMatrix> getOutputs = new DataProvider<>(
                        new Object[]{i, numImages / threads}) {
                    @Override
                    public CPUMatrix getData(Object[] args) {
                        Character trainValue = getIndex("C:\\oroarmor\\numberID\\train\\labels.txt",
                                (Integer) (globalArgs[0]) * (Integer) (globalArgs[1]) + (Integer) (args[0]));
                        CPUMatrix output = new CPUMatrix(10, 1);
                        output.setValue(Integer.parseInt(trainValue + ""), 0, 1);
                        return output;
                    }
                };

                Trainer<CPUMatrix> trainer = new Trainer<>(getInputs, getOutputs, numberIDNetwork,
                        new TotalError(0.01));

                Thread thread = new Thread(trainer);
                trainingThreads[i] = thread;
            }
            for (Thread thread : trainingThreads) {
                thread.start();
            }

            try {
                for (Thread thread : trainingThreads) {
                    thread.join();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            System.out.println(repeats + " " + (System.currentTimeMillis() - start) / (1000f * (repeats + 1)));
            test();
        }
        System.out.println(numberIDNetwork.getTrainingAttempts());
        System.out.println((System.currentTimeMillis() - start) / 1000f + " total seconds");
        NetworkSaver.saveNetworkToFile(numberIDNetwork, "C:\\oroarmor\\numberID\\", "numberIDNetwork.nn");
    }

    public void test() {
        Tester.numCorrect = 0;
        int threads = 8;
        Thread[] testThreads = new Thread[threads];

        double numImages = 10000;
        DecimalFormat format = new DecimalFormat("#.##");

        for (int i = 0; i < threads; i++) {

            DataProvider<CPUMatrix> getInputs = new DataProvider<>(new Object[]{i, (int)numImages / threads}) {
                @Override
                public CPUMatrix getData(Object[] args) {
                    return getImageData(loadImage("C:\\oroarmor\\numberID\\test\\images\\"
                            + ((Integer) (globalArgs[0]) * (Integer) (globalArgs[1])
                            + (Integer) (args[0]))
                            + ".png"));
                }
            };

            DataProvider<CPUMatrix> getOutputs = new DataProvider<>(new Object[]{i, (int)numImages / threads}) {
                @Override
                public CPUMatrix getData(Object[] args) {
                    Character trainValue = getIndex("C:\\oroarmor\\numberID\\test\\labels.txt",
                            (Integer) (globalArgs[0]) * (Integer) (globalArgs[1]) + (Integer) (args[0]));
                    CPUMatrix output = new CPUMatrix(10, 1);
                    output.setValue(Integer.parseInt(trainValue + ""), 0, 1);
                    return output;
                }
            };

            Tester<CPUMatrix> tester = new Tester<>(getInputs, getOutputs, numberIDNetwork, 0.5);

            Thread thread = new Thread(tester);
            testThreads[i] = thread;
        }
        for (Thread thread : testThreads) {
            thread.start();
        }

        try {
            for (Thread thread : testThreads) {
                thread.join();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(format.format(Tester.numCorrect / numImages * 100) + "%");
    }

    public Character getIndex(String textFilePath, int index) {
        FileInputStream textFile;
        try {
            textFile = new FileInputStream(new File(textFilePath));
            long skip = textFile.skip(index);
            Character value = (char) textFile.read();
            textFile.close();
            return value;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
