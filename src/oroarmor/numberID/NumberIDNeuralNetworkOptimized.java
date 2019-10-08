package oroarmor.numberID;

import java.io.File;
import java.io.FileInputStream;

import oroarmor.neuralnetwork.layer.FeedFowardLayer;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.SoftMaxFunction;
import oroarmor.neuralnetwork.network.NetworkSaver;
import oroarmor.neuralnetwork.network.NeuralNetwork;
import oroarmor.neuralnetwork.training.GetData;
import oroarmor.neuralnetwork.training.Tester;
import oroarmor.neuralnetwork.training.Trainer;
import oroarmor.neuralnetwork.training.models.TotalError;
import processing.core.PApplet;
import processing.core.PImage;

public class NumberIDNeuralNetworkOptimized extends PApplet {

	boolean reset = false;
	NeuralNetwork numberIDNetwork;

	public static int trainingImages = 60000;
	public static int testingImages = 10000;
	
	public static PImage[] trainImages = new PImage[trainingImages];
	public static int[] trainNumbers = new int[trainingImages];

	public static PImage[] testImages = new PImage[testingImages];
	public static int[] testNumbers = new int[testingImages];

	

	public static void main(String[] args) {
		PApplet.main("oroarmor.numberID.NumberIDNeuralNetworkOptimized");
	}

	public Matrix getImageData(String string) {
		return getImageData(loadImage(string));
	}

	public Matrix getImageData(PImage image) {
		image.loadPixels();
		double[][] matrixArray = new double[image.height * image.width][1];
		for (int i = 0; i < matrixArray.length; i++) {
			matrixArray[i] = new double[] { (double) brightness(image.pixels[i]) / 255f };
		}
		return new Matrix(matrixArray);
	}

	public void settings() {
		size(280, 400);
	}

	public void setup() {
		background(0);
		noStroke();
		numberIDNetwork = NetworkSaver.loadNetworkFromFile("C:\\oroarmor\\numberID\\",
				"NumberIDNeuralNetworkOptimized.nn");

		if (numberIDNetwork == null || reset) {
			numberIDNetwork = new NeuralNetwork(28 * 28);
			numberIDNetwork.addLayer(new FeedFowardLayer(64));
			numberIDNetwork.addLayer(new FeedFowardLayer(32));
			numberIDNetwork.addLayer(new FeedFowardLayer(16));
			numberIDNetwork.addLayer(new FeedFowardLayer(16));
			numberIDNetwork.addLayer(new FeedFowardLayer(10));
		}

		System.out.println("Setup values");
//		long temp = System.currentTimeMillis();
		setUpImagesAndValues("C:\\oroarmor\\numberID\\train\\", "labels.txt", trainingImages, trainImages,
				trainNumbers);
		setUpImagesAndValues("C:\\oroarmor\\numberID\\test\\", "labels.txt", testingImages, testImages, testNumbers);
//		System.out.println((System.currentTimeMillis() - temp) / 1000);
		
		System.out.println("train+test");

		test();
		train();
//		test();

//		noLoop();
//		strokeWeight(28);
	}

	public void setUpImagesAndValues(String path, String labelName, int numImages, PImage[] imageArray,
			int[] numberArray) {
		for (int i = 0; i < numImages; i++) {
			imageArray[i] = loadImage(path + "images\\" + i + ".png");
			numberArray[i] = Integer.parseInt(getIndex(path + labelName, i) + "");
		}
	}

	public void draw() {
		if (mousePressed) {
			stroke(255, 255 / 2);
			for (int i = 0; i < 5; i++) {
				strokeWeight(25 - i * 5);
				line(mouseX, mouseY, pmouseX, pmouseY);
			}
		}
		fill(255);
		rect(0, 280, 280, 120);

		PImage imageFromScreen = new PImage(280, 280);
		imageFromScreen.loadPixels();

		imageFromScreen.copy(g.copy(), 0, 0, 280, 280, 0, 0, 280, 280);

		imageFromScreen.resize(28, 28);
		imageFromScreen.updatePixels();

		Matrix outputs = numberIDNetwork.feedFoward(getImageData(imageFromScreen)).applyFunction(new SoftMaxFunction());

		noStroke();
		for (int i = 0; i < 10; i++) {
			fill((float) (1 - outputs.getValue(i, 0)) * 255f, (float) (outputs.getValue(i, 0)) * 255f, 0);
			rect(i * 280 / 10, 280, 28, 120);
			fill(0);
			text(i + ": \n" + outputs.getValue(i, 0), i * 28, 300);
		}
	}

	public void keyPressed() {
		if (key == 'c') {
			background(0);
		}
	}

	public void train() {
		long start = System.currentTimeMillis();
		int numImages = trainingImages;

		for (int repeats = 0; repeats < 50; repeats++) {
			int threads = 16;
			Thread[] trainingThreads = new Thread[threads];

			for (int i = 0; i < threads; i++) {

				GetData getInputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
					public Matrix getData(String[] args) {
						int i = Integer.parseInt(args[0]);
						int threadIndex = Integer.parseInt(globalArgs[1]) * Integer.parseInt(globalArgs[0]);
						Matrix images = getImageData(NumberIDNeuralNetworkOptimized.trainImages[threadIndex + i]);
						return images;
					}
				};

				GetData getOutputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
					public Matrix getData(String[] args) {
						int i = Integer.parseInt(args[0]);
						int threadIndex = Integer.parseInt(globalArgs[1]) * Integer.parseInt(globalArgs[0]);
						Matrix output = new Matrix(10, 1);
						output.setValue(NumberIDNeuralNetworkOptimized.trainNumbers[i + threadIndex], 0, 1);
						return output;
					}
				};
				Trainer trainer = new Trainer(getInputs, getOutputs, numberIDNetwork, new TotalError(0.01));
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
			System.out.println(repeats + " " + ((System.currentTimeMillis() - start) / (1000f * (repeats + 1))));
		}
		System.out.println(numberIDNetwork.getTrainingAttemps());
		System.out.println((System.currentTimeMillis() - start) / 1000f + " total seconds");
		NetworkSaver.saveNetworkToFile(numberIDNetwork, "NumberIDNeuralNetworkOptimized.nn",
				"C:\\oroarmor\\numberID\\");
	}

	public void test() {
		Tester.numCorrect = 0;
		int threads = 8;
		Thread[] testThreads = new Thread[threads];

		int numImages = testingImages;

		for (int i = 0; i < threads; i++) {

			GetData getInputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
				public Matrix getData(String[] args) {
					int i = Integer.parseInt(args[0]);
					int threadIndex = Integer.parseInt(globalArgs[1]) * Integer.parseInt(globalArgs[0]);
					Matrix images = getImageData(NumberIDNeuralNetworkOptimized.testImages[threadIndex + i]);
					return images;
				}
			};

			GetData getOutputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
				public Matrix getData(String[] args) {
					int i = Integer.parseInt(args[0]);
					int threadIndex = Integer.parseInt(globalArgs[1]) * Integer.parseInt(globalArgs[0]);
					Matrix output = new Matrix(10, 1);
					output.setValue(NumberIDNeuralNetworkOptimized.testNumbers[i + threadIndex], 0, 1);
					return output;
				}
			};

			Tester tester = new Tester(getInputs, getOutputs, numberIDNetwork);

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
		System.out.println(Tester.numCorrect);
	}

	public Character getIndex(String textFilePath, int index) {
		FileInputStream textFile = null;
		try {
			textFile = new FileInputStream(new File(textFilePath));
			textFile.skip(index);
			Character value = (char) textFile.read();
			textFile.close();
			return value;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

//	public void mouseClicked() {
//		setup();
//	}
}
