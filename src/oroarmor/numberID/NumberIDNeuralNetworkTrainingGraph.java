package oroarmor.numberID;

import java.io.File;
import java.io.FileInputStream;

import oroarmor.neuralnetwork.layer.FeedFowardLayer;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.network.NetworkSaver;
import oroarmor.neuralnetwork.network.NeuralNetwork;
import oroarmor.neuralnetwork.training.GetData;
import oroarmor.neuralnetwork.training.Tester;
import oroarmor.neuralnetwork.training.Trainer;
import oroarmor.neuralnetwork.training.models.TotalError;
import processing.core.PApplet;
import processing.core.PImage;

public class NumberIDNeuralNetworkTrainingGraph extends PApplet {

	NeuralNetwork numberIDNetwork;
	PImage randomImage;

	int totalTrains = 100;

	public static void main(String[] args) {
		PApplet.main("oroarmor.numberID.NumberIDNeuralNetworkTrainingGraph");
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

		System.out.println("Creating network...");

		numberIDNetwork = new NeuralNetwork(28 * 28);
		numberIDNetwork.addLayer(new FeedFowardLayer(64));
		numberIDNetwork.addLayer(new FeedFowardLayer(32));
		numberIDNetwork.addLayer(new FeedFowardLayer(16));
		numberIDNetwork.addLayer(new FeedFowardLayer(16));
		numberIDNetwork.addLayer(new FeedFowardLayer(10));
		System.out.println("Training...");
		train();
	}

	public void draw() {
		int[] numCorrect = new int[totalTrains];
		for (int i = 0; i < totalTrains; i++) {
			NeuralNetwork currentLevel = NetworkSaver.loadNetworkFromFile("C:\\oroarmor\\numberID\\trainingGraph\\",
					"numberIDNetwork_" + (i + 1) + "Trains.nn");
			numCorrect[i] = test(currentLevel);
		}
		println(numCorrect);
		noLoop();
	}

	public void train() {
		long start = System.currentTimeMillis();
		int numImages = 60000;
		for (int repeats = 0; repeats < totalTrains; repeats++) {
			int threads = 36;
			Thread[] trainingThreads = new Thread[threads];
			for (int i = 0; i < threads; i++) {

				GetData getInputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
					public Matrix getData(String[] args) {
						Matrix images = getImageData(loadImage("C:\\oroarmor\\numberID\\train\\images\\"
								+ (Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
										+ Integer.parseInt(args[0]))
								+ ".png"));
						return images;
					}
				};

				GetData getOutputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
					public Matrix getData(String[] args) {
						Character trainValue = getIndex("C:\\oroarmor\\numberID\\train\\labels.txt",
								Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
										+ Integer.parseInt(args[0]));
						Matrix output = new Matrix(10, 1);
						output.setValue(Integer.parseInt(trainValue + ""), 0, 1);
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
			NetworkSaver.saveNetworkToFile(numberIDNetwork, "numberIDNetwork_" + (repeats + 1) + "Trains.nn",
					"C:\\oroarmor\\numberID\\trainingGraph\\");
			System.out.println(repeats + " " + ((System.currentTimeMillis() - start) / (1000f * (repeats + 1))));
		}
		System.out.println(numberIDNetwork.getTrainingAttemps());
		System.out.println((System.currentTimeMillis() - start) / 1000f + " total seconds");

	}

	public int test(NeuralNetwork network) {
		Tester.numCorrect = 0;
		int threads = 36;
		Thread[] testThreads = new Thread[threads];

		int numImages = 10000;

		for (int i = 0; i < threads; i++) {

			GetData getInputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
				public Matrix getData(String[] args) {
					Matrix images = getImageData(loadImage("C:\\oroarmor\\numberID\\test\\images\\"
							+ (Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
									+ Integer.parseInt(args[0]))
							+ ".png"));
					return images;
				}
			};

			GetData getOutputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
				public Matrix getData(String[] args) {
					Character trainValue = getIndex("C:\\oroarmor\\numberID\\test\\labels.txt",
							Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
									+ Integer.parseInt(args[0]));
					Matrix output = new Matrix(10, 1);
					output.setValue(Integer.parseInt(trainValue + ""), 0, 1);
					return output;
				}
			};

			Tester tester = new Tester(getInputs, getOutputs, network);

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
		return Tester.numCorrect;
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
}
