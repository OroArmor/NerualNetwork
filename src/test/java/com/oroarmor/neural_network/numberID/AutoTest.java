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
	boolean reset = false;

	public static int numImages = 16000;
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
		encoder = (AutoEncoder<CPUMatrix>) NetworkSaver.loadNetworkFromFile("C:\\oroarmor\\", "test.nn");
		if (encoder == null || reset) {
			encoder = new AutoEncoder<>(28 * 28, 2);
			encoder.addLayer(new FeedForwardLayer<>(128, Matrix.MatrixType.CPU));
			encoder.addLayer(new FeedForwardLayer<>(64, Matrix.MatrixType.CPU));
			encoder.addLayer(new FeedForwardLayer<>(10,Matrix.MatrixType.CPU));
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
				DataProvider<CPUMatrix> getData = new DataProvider<>(new Object[] { thread, numImages / numThreads}) {
					@Override
					public CPUMatrix getData(Object[] args) {
						int image = (Integer)(args[0]);
						int threadIndex = (Integer)(globalArgs[1]) * (Integer)(globalArgs[0]);
						return images[image + threadIndex];
					}
				};
				Trainer<CPUMatrix> trainer = new Trainer<>(getData, getData, encoder, new TotalError(0.01));
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
			System.out.println(System.currentTimeMillis() - millis);
		}

		NetworkSaver.saveNetworkToFile(encoder, "C:\\oroarmor\\", "test.nn");
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
		CPUMatrix test = Matrix.randomMatrix(Matrix.MatrixType.CPU, 10, 1, new Random(), 0, 1);
		assert test != null;
		test.applyFunction(new SoftMaxFunction());

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
