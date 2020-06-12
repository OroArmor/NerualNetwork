import oroarmor.neuralnetwork.layer.FeedFowardLayer;
import oroarmor.neuralnetwork.matrix.CPUMatrix;
import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.network.NeuralNetwork;
import oroarmor.neuralnetwork.training.models.TotalError;
import processing.core.PApplet;

public class XORProblem extends PApplet {

	public static void main(String[] args) {
		PApplet.main("XORProblem");
	}

	NeuralNetwork xornn = new NeuralNetwork(2);
	double[] input1 = { 1, 1 };
	double[] input2 = { 0, 1 };
	double[] input3 = { 1, 0 };

	double[] input4 = { 0, 0 };
	double[] output1 = { 0 };
	double[] output2 = { 1 };
	double[] output3 = { 1 };

	double[] output4 = { 0 };
	Matrix inputs1 = new CPUMatrix(input1, 2, 1);
	Matrix inputs2 = new CPUMatrix(input2, 2, 1);
	Matrix inputs3 = new CPUMatrix(input3, 2, 1);

	Matrix inputs4 = new CPUMatrix(input4, 2, 1);
	double[][] output = { { 0 } };

	Matrix[] inputs = { inputs1, inputs2, inputs3, inputs4 };
	Matrix outputs1 = new CPUMatrix(output1, 1, 1);
	Matrix outputs2 = new CPUMatrix(output2, 1, 1);
	Matrix outputs3 = new CPUMatrix(output3, 1, 1);

	Matrix outputs4 = new CPUMatrix(output4, 1, 1);

	Matrix[] outputs = { outputs1, outputs2, outputs3, outputs4 };

	int trains = 0;

	@Override
	public void draw() {
		background(255);
		for (int i = 0; i < 1000; i++) {
			for (int j = 0; j < inputs.length; j++) {
				xornn.train(inputs[j], outputs[j], new TotalError(0.01));
				trains++;
			}
		}

		float res = 100;
		for (int i = 0; i < width; i += width / res) {
			for (int j = 0; j < height; j += height / res) {
				double[] currentInput = { (double) i / width, (double) j / height };
				double output = xornn.feedFoward(new CPUMatrix(currentInput, 2, 1)).getValue(0, 0);
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
		xornn.addLayer(new FeedFowardLayer(4));
		xornn.addLayer(new FeedFowardLayer(1));
		noStroke();
	}
}
