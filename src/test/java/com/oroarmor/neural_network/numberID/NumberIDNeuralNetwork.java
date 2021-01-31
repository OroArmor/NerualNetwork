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

public class NumberIDNeuralNetwork extends PApplet {

	boolean reset = false;
	NeuralNetwork numberIDNetwork;
	PImage randomImage;

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
		size(280, 400);
	}

	public void setup() {
		background(0);
		noStroke();
		numberIDNetwork = NetworkSaver.loadNetworkFromFile("C:\\oroarmor\\numberID\\", "numberIDNetwork.nn");

		if (numberIDNetwork == null || reset) {
			numberIDNetwork = new NeuralNetwork(28 * 28);
			numberIDNetwork.addLayer(new FeedFowardLayer(64));
			numberIDNetwork.addLayer(new FeedFowardLayer(32));
			numberIDNetwork.addLayer(new FeedFowardLayer(16));
			numberIDNetwork.addLayer(new FeedFowardLayer(16));
			numberIDNetwork.addLayer(new FeedFowardLayer(10));
		}
//		test();
		train(); 
		test();
//		noLoop();
//		strokeWeight(28);
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
		int numImages = 60000;
		
		
		
		for (int repeats = 0; repeats < 5; repeats++) {
			int threads = 8;
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
			System.out.println(repeats + " " + ((System.currentTimeMillis() - start) / (1000f * (repeats + 1))));
		}
		System.out.println(numberIDNetwork.getTrainingAttemps());
		System.out.println((System.currentTimeMillis() - start) / 1000f + " total seconds");
		NetworkSaver.saveNetworkToFile(numberIDNetwork, "numberIDNetwork.nn", "C:\\oroarmor\\numberID\\");
	}

	public void test() {
		Tester.numCorrect = 0;
		int threads = 8;
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
