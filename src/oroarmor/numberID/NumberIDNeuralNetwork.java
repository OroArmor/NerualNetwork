package oroarmor.numberID;

import java.io.File;
import java.io.FileInputStream;

import oroarmor.neuralnetwork.layer.FeedFowardLayer;
import oroarmor.neuralnetwork.layer.SoftMaxLayer;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.network.NetworkSaver;
import oroarmor.neuralnetwork.network.NeuralNetwork;
import oroarmor.neuralnetwork.training.GetData;
import oroarmor.neuralnetwork.training.Trainer;
import oroarmor.neuralnetwork.training.models.TotalError;
import processing.core.PApplet;
import processing.core.PImage;

public class NumberIDNeuralNetwork extends PApplet {

	boolean reset = false;
	NeuralNetwork numberIDNetwork;

	public static void main(String[] args) {
		PApplet.main("oroarmor.numberID.NumberIDNeuralNetwork");
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
		size(280, 280);
	}

	public void setup() {
		int randomID = (int) random(0, 10000);
		PImage randomImage = loadImage("C:\\oroarmor\\numberID\\test\\images\\" + randomID + ".png");
		Character randomValue = getIndex("C:\\oroarmor\\numberID\\test\\labels.txt", randomID);
		image(randomImage, 0, 0, 280, 280);
		numberIDNetwork = NetworkSaver.loadNetworkFromFile("C:\\oroarmor\\numberID\\", "numberIDNetwork.nn");

		if (numberIDNetwork == null || reset) {
			numberIDNetwork = new NeuralNetwork(28 * 28);

			numberIDNetwork.addLayer(new FeedFowardLayer(64));
			numberIDNetwork.addLayer(new FeedFowardLayer(32));
			numberIDNetwork.addLayer(new FeedFowardLayer(16));
			numberIDNetwork.addLayer(new FeedFowardLayer(16));
			numberIDNetwork.addLayer(new SoftMaxLayer(10));
		}
		Matrix first = numberIDNetwork.feedFoward(getImageData(randomImage)).print();

		Thread[] trainingThreads = new Thread[4];

		int numImages = 6000;
		int threads = 3;
		long start = System.currentTimeMillis();

		for (int i = 0; i < 4; i++) {

			GetData getInputs = new GetData(new String[] { i + "", numImages / threads + "" }) {
				public Matrix getData(String[] args) {
					Matrix images = getImageData(loadImage("C:\\oroarmor\\numberID\\train\\images\\"
							+ (Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
									+ Integer.parseInt(args[0]))
							+ ".png"));
					return images;
				}
			};

			GetData getOutputs = new GetData(new String[] { i + "", numImages / threads + ""  }) {
				public Matrix getData(String[] args) {
					Character trainValue = getIndex("C:\\oroarmor\\numberID\\train\\labels.txt",
							Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1]) + Integer.parseInt(args[0]));
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

		numberIDNetwork.feedFoward(getImageData(randomImage)).subtractMatrix(first).print();
		System.out.println(numberIDNetwork.getTrainingAttemps());
		System.out.println(System.currentTimeMillis() - start);
//		NetworkSaver.saveNetworkToFile(numberIDNetwork, "numberIDNetwork.nn", "C:\\oroarmor\\numberID\\");
	}

	public void draw() {

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
