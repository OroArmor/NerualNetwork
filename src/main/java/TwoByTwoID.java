import oroarmor.neuralnetwork.layer.FeedFowardLayer;
import oroarmor.neuralnetwork.layer.KeepPositiveLayer;
import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.Matrix.MatrixType;
import oroarmor.neuralnetwork.network.NetworkSaver;
import oroarmor.neuralnetwork.network.NeuralNetwork;
import oroarmor.neuralnetwork.training.models.TotalError;
import processing.core.PApplet;

public class TwoByTwoID extends PApplet {

	public static void main(String[] args) {
		PApplet.main("TwoByTwoID");
	}

	DisposeHandler dh;

	NeuralNetwork twobytwonn;
	Matrix[] inputs;

	Matrix[] outputs;

	boolean reset = true;

	@Override
	public void draw() {
		background(255);
		for (int i = 0; i < 10000; i++) {
			for (int j = 0; j < inputs.length; j++) {
				twobytwonn.train(inputs[j], outputs[j], new TotalError(0.01));
			}
		}
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 2; j++) {
				drawInputs(inputs[i + j * 4], i * width / 4, j * height / 2, width / 4, height / 2, i + j * 4);
			}
		}
		fill(0);
		text(twobytwonn.getTrainingAttemps(), 200, 380);
	}

	void drawInputs(Matrix inputs, float x, float y, float w, float h, int oIndex) {

		Matrix outputs = twobytwonn.feedFoward(inputs);
		pushMatrix();
		translate(x + w / 2f, y + h / 4f);
		scale(0.8f);

		double[] output = outputs.getValues();

		int index = 0;
		double max = Double.MIN_VALUE;

		for (int i = 0; i < output.length; i++) {
			if (output[i] > max) {
				index = i;
				max = output[i];
			}
		}

		String what = "error";

		switch (index) {
			case 0:
				what = "solid";
				break;
			case 1:
				what = "diagonal";
				break;
			case 2:
				what = "horizontal";
				break;
			case 3:
				what = "vertical";
				break;
			default:
				println(index);
		}

		String actual = "";

		if (oIndex > 5) {
			actual = "vertical";
		} else if (oIndex > 3) {
			actual = "horizontal";
		} else if (oIndex > 1) {
			actual = "diagonal";
		} else {
			actual = "solid";
		}
		fill(255, 0, 0);
		if (actual.equals(what)) {
			fill(0, 255, 0);
		}

		rect(0 - w / 2f, 0 - h / 4f, w, h);
		fill(0);
		textSize(20);
		text(what, 0, h / 2f);

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				fill((float) inputs.getValue(i * 2 + j, 0) * 255f);

				rect((i - 1) * w / 2f, (j - 1) * h / 4f, w / 2f, h / 4f);

				fill(255, 0, 0);
				text(i * 2 + j, (i - 1) * w / 2f + w / 4f, (j - 1) * h / 4f + h / 8f);
			}
		}

		popMatrix();
	}

	@Override
	public void settings() {
		size(400, 400);
	}

	@Override
	public void setup() {

		double[][][] ins = { { { 0, 0, 0, 0 } }, { { 1, 1, 1, 1 } },

				{ { 1, 0, 0, 1 } }, { { 0, 1, 1, 0 } },

				{ { 0, 1, 0, 1 } }, { { 1, 0, 1, 0 } },

				{ { 0, 0, 1, 1 } }, { { 1, 1, 0, 0 } } };

		double[][] sols = { { 1, 0, 0, 0 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 },
				{ 0, 0, 1, 0 }, { 0, 0, 0, 1 }, { 0, 0, 0, 1 } };
		inputs = new Matrix[8];
		outputs = new Matrix[8];
		for (int i = 0; i < sols.length; i++) {
			inputs[i] = new CPUMatrix(ins[i][0], 4, 1);
			outputs[i] = new CPUMatrix(sols[i], sols[i].length, 1);
		}

		twobytwonn = NetworkSaver.loadNetworkFromFile(System.getProperty("user.dir") + "/src/data/savedNetworks/2x2/",
				"twoXtwonn.nn");

		if (twobytwonn == null || reset) {
			twobytwonn = new NeuralNetwork(4);
			twobytwonn.addLayer(new FeedFowardLayer(4, MatrixType.CPU));
			twobytwonn.addLayer(new FeedFowardLayer(4, MatrixType.CPU));
			twobytwonn.addLayer(new KeepPositiveLayer(8, MatrixType.CPU));
			twobytwonn.addLayer(new FeedFowardLayer(4, MatrixType.CPU));
		}
		System.out.println("Feed Foward");
		for (Matrix input : inputs) {
			twobytwonn.feedFoward(input);
		}
//		noStroke();
		textAlign(CENTER, CENTER);

		dh = new DisposeHandler();
		registerMethod("dispose", dh);
//		dh.register(this);
	}

	public class DisposeHandler {
		public DisposeHandler() {
		}

		public void dispose() {
			NetworkSaver.saveNetworkToFile(twobytwonn, "twoXtwonn.nn",
					System.getProperty("user.dir") + "/src/data/savedNetworks/2x2/");
			System.out.println("Network Saved");

		}

		public void register(PApplet p) {
			p.registerMethod("dispose", this);

		}
	}

}
