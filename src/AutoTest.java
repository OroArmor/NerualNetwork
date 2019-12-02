import java.io.File;
import java.util.Random;

import oroarmor.neuralnetwork.layer.FeedFowardLayer;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.network.AutoEncoder;
import oroarmor.neuralnetwork.network.NetworkSaver;
import oroarmor.neuralnetwork.training.GetData;
import oroarmor.neuralnetwork.training.Trainer;
import oroarmor.neuralnetwork.training.models.TotalError;

import processing.core.PApplet;

public class AutoTest extends PApplet {
	AutoEncoder encoder;
	boolean reset = true;

	public static int numImages = 15000;
	public static Matrix[] images = new Matrix[numImages];

	public static void main(String[] args) {
		System.out.print("Loading Data");
		long startTime = System.currentTimeMillis();
		byte[] dataBytes = PApplet.loadBytes(new File(
				System.getProperty("user.dir") + "/src/data/numberID/train/train-images-idx3-ubyte-uncompressed"));
		for (int k = 0; k < numImages; k++) {
			double[] imageDoubles = new double[28 * 28];
			for (byte i = 0; i < 28; i++) {
				for (byte j = 0; j < 28; j++) {
					byte value = dataBytes[16 + j * 28 + i + (k) * 28 * 28];
					imageDoubles[i * 28 + j] = ((double) ((value >= 0) ? value : value + 255d)) / 255d;
				}
			}
			images[k] = new Matrix(imageDoubles);
			if (k % 1200 == 0) {
				System.out.print(".");
			}
		}
		System.out.println("\nData loaded in: " + ((System.currentTimeMillis() - startTime) / 1000) + " seconds");
		PApplet.main("AutoTest");
	}

	public void setup() {
		encoder = (AutoEncoder) NetworkSaver.loadNetworkFromFile("C:\\oroarmor\\", "test.nn");
		if (encoder == null || reset) {
			encoder = new AutoEncoder(28 * 28, 2);

			encoder.addLayer(new FeedFowardLayer(128));
			encoder.addLayer(new FeedFowardLayer(64));
			encoder.addLayer(new FeedFowardLayer(32));
			encoder.addLayer(new FeedFowardLayer(64));
			encoder.addLayer(new FeedFowardLayer(128));
			encoder.addLayer(new FeedFowardLayer(28 * 28));
		}

		System.out.println(encoder.trains);

		int numThreads = 16;
		int numRepeats = 0;

		Thread[] threads = new Thread[numThreads];

		for (int repeat = 0; repeat < numRepeats; repeat++) {
			long millis = System.currentTimeMillis();
			for (int thread = 0; thread < numThreads; thread++) {

				GetData getData = new GetData(new String[] { thread + "", numImages / numThreads + "" }) {
					public Matrix getData(String[] args) {
						int image = Integer.parseInt(args[0]);
						int threadIndex = Integer.parseInt(globalArgs[1]) * Integer.parseInt(globalArgs[0]);
						return images[image + threadIndex];
					}
				};
				Trainer trainer = new Trainer(getData, getData, encoder, new TotalError(0.01));
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
			System.out.println((System.currentTimeMillis() - millis) / 1000);
		}

		System.out.println(encoder.trains);
		NetworkSaver.saveNetworkToFile(encoder, "test.nn", "C:\\oroarmor\\");
		noStroke();

//		mouseClicked();
	}

	public void settings() {
		size(280, 280);
	}

	public void draw() {
		frameRate(1);
		background(0, 0, 0);
		Matrix test = new Matrix(32, 1);
		Random random = new Random();
		random.setSeed((long) random(-1000, 1000));
		test.randomize(random, 0, 1);
		test.print();
		double[][] values = encoder.feedFoward(test).getValues();
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				fill((int) (values[i * 28 + j][0] * 255));
				rect(i * 10, j * 10, 10, 10);
			}
		}
		System.out.println("New Frame");
	}
}
