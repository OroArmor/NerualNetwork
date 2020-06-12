package oroarmor.numberID;

import java.io.File;
import java.io.FileInputStream;

import oroarmor.neuralnetwork.layer.FeedFowardLayer;
import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.function.SoftMaxFunction;
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
	NeuralNetwork<CPUMatrix> numberIDNetwork;
	PImage randomImage;

	public static void main(String[] args) {
		PApplet.main("oroarmor.numberID.NumberIDNeuralNetwork");
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
		size(280, 400);
	}

	@Override
	public void setup() {
		background(0);
		noStroke();
		numberIDNetwork = NetworkSaver.<CPUMatrix>loadNetworkFromFile("C:\\oroarmor\\numberID\\", "numberIDNetwork.nn");

		if (numberIDNetwork == null || reset) {
			numberIDNetwork = new NeuralNetwork<CPUMatrix>(28 * 28);
			numberIDNetwork.addLayer(new FeedFowardLayer<CPUMatrix>(64));
			numberIDNetwork.addLayer(new FeedFowardLayer<CPUMatrix>(32));
			numberIDNetwork.addLayer(new FeedFowardLayer<CPUMatrix>(16));
			numberIDNetwork.addLayer(new FeedFowardLayer<CPUMatrix>(16));
			numberIDNetwork.addLayer(new FeedFowardLayer<CPUMatrix>(10));
		}
		test();
		train();
		test();

//		try {
//			Runtime.getRuntime().exec("shutdown -h");
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}

//		noLoop();
//		strokeWeight(28);
	}

	@Override
	public void draw() {
//		background(0);
		if (mousePressed) {
			stroke(255, 255 / 2);
			for (int i = 0; i < 5; i++) {
				strokeWeight(25 - i * 5);
				line(mouseX, mouseY, pmouseX, pmouseY);
			}
		}
		fill(255);
		noStroke();
		rect(0, 280, 280, 120);

		PImage myTest = new PImage(280, 280);
		myTest.loadPixels();

		myTest.copy(g.copy(), 0, 0, 280, 280, 0, 0, 280, 280);

		myTest.resize(28, 28);
		myTest.updatePixels();

		CPUMatrix outputs = numberIDNetwork.feedFoward(getImageData(myTest)).applyFunction(new SoftMaxFunction());

		noStroke();
		for (int i = 0; i < 10; i++) {
//			fill(255,0,0);
//			if(i == max) {
//				fill(0,255,0);
//			}
			fill((float) (1 - outputs.getValue(i, 0)) * 255f, (float) outputs.getValue(i, 0) * 255f, 0);
			rect(i * 280 / 10, 280, 28, 120);

			fill(0);
			text(i + ": \n" + outputs.getValue(i, 0), i * 28, 300);

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
		for (int repeats = 0; repeats < 10; repeats++) {
			int threads = 12;
			Thread[] trainingThreads = new Thread[threads];

			int numImages = 60000;

			for (int i = 0; i < threads; i++) {

				GetData<CPUMatrix> getInputs = new GetData<CPUMatrix>(
						new String[] { i + "", numImages / threads + "" }) {
					@Override
					public CPUMatrix getData(String[] args) {
						CPUMatrix images = getImageData(loadImage("C:\\oroarmor\\numberID\\train\\images\\"
								+ (Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
										+ Integer.parseInt(args[0]))
								+ ".png"));
						return images;
					}
				};

				GetData<CPUMatrix> getOutputs = new GetData<CPUMatrix>(
						new String[] { i + "", numImages / threads + "" }) {
					@Override
					public CPUMatrix getData(String[] args) {
						Character trainValue = getIndex("C:\\oroarmor\\numberID\\train\\labels.txt",
								Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
										+ Integer.parseInt(args[0]));
						CPUMatrix output = new CPUMatrix(10, 1);
						output.setValue(Integer.parseInt(trainValue + ""), 0, 1);
						return output;
					}
				};

				Trainer<CPUMatrix> trainer = new Trainer<CPUMatrix>(getInputs, getOutputs, numberIDNetwork,
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

			GetData<CPUMatrix> getInputs = new GetData<CPUMatrix>(new String[] { i + "", numImages / threads + "" }) {
				@Override
				public CPUMatrix getData(String[] args) {
					CPUMatrix images = getImageData(loadImage("C:\\oroarmor\\numberID\\test\\images\\"
							+ (Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
									+ Integer.parseInt(args[0]))
							+ ".png"));
					return images;
				}
			};

			GetData<CPUMatrix> getOutputs = new GetData<CPUMatrix>(new String[] { i + "", numImages / threads + "" }) {
				@Override
				public CPUMatrix getData(String[] args) {
					Character trainValue = getIndex("C:\\oroarmor\\numberID\\test\\labels.txt",
							Integer.parseInt(globalArgs[0]) * Integer.parseInt(globalArgs[1])
									+ Integer.parseInt(args[0]));
					CPUMatrix output = new CPUMatrix(10, 1);
					output.setValue(Integer.parseInt(trainValue + ""), 0, 1);
					return output;
				}
			};

			Tester<CPUMatrix> tester = new Tester<CPUMatrix>(getInputs, getOutputs, numberIDNetwork);

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
