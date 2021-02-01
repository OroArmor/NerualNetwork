package com.oroarmor.neural_network;

import com.oroarmor.neural_network.layer.FeedForwardLayer;
import com.oroarmor.neural_network.matrix.CPUMatrix;
import com.oroarmor.neural_network.matrix.Matrix.MatrixType;
import com.oroarmor.neural_network.network.NeuralNetwork;
import com.oroarmor.neural_network.training.models.TotalError;
import processing.core.PApplet;

public class XORProblem extends PApplet {
    NeuralNetwork<CPUMatrix> xornn = new NeuralNetwork<>(2);
    double[] input1 = {1, 1};
    double[] input2 = {0, 1};
    double[] input3 = {1, 0};
    double[] input4 = {0, 0};
    double[] output1 = {0};
    double[] output2 = {1};
    double[] output3 = {1};
    double[] output4 = {0};
    CPUMatrix inputs1 = new CPUMatrix(input1, 2, 1);
    CPUMatrix inputs2 = new CPUMatrix(input2, 2, 1);
    CPUMatrix inputs3 = new CPUMatrix(input3, 2, 1);
    CPUMatrix inputs4 = new CPUMatrix(input4, 2, 1);
    CPUMatrix[] inputs = {inputs1, inputs2, inputs3, inputs4};
    CPUMatrix outputs1 = new CPUMatrix(output1, 1, 1);
    CPUMatrix outputs2 = new CPUMatrix(output2, 1, 1);
    CPUMatrix outputs3 = new CPUMatrix(output3, 1, 1);
    CPUMatrix outputs4 = new CPUMatrix(output4, 1, 1);
    CPUMatrix[] outputs = {outputs1, outputs2, outputs3, outputs4};
    int trains = 0;

    public static void main(String[] args) {
        PApplet.main("com.oroarmor.neural_network.XORProblem");
    }

    @Override
    public void draw() {
        background(255);
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < inputs.length; j++) {
                xornn.train(inputs[j], outputs[j], new TotalError(0.1));
                trains++;
            }
        }

        float res = 100;
        for (int i = 0; i < width; i += width / res) {
            for (int j = 0; j < height; j += height / res) {
                double[] currentInput = {(double) i / width, (double) j / height};
                double output = xornn.feedForward(new CPUMatrix(currentInput, 2, 1)).getValue(0, 0);
                fill((float) output * 255);
                rect(i, j, width / res, height / res);
            }
        }
    }

    @Override
    public void settings() {
        size(400, 400);
    }

    @Override
    public void setup() {
        xornn.addLayer(new FeedForwardLayer<>(4, MatrixType.CPU));
        xornn.addLayer(new FeedForwardLayer<>(1, MatrixType.CPU));
        noStroke();
    }
}
