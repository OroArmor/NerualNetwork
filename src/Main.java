import oroarmor.layer.FeedFowardLayer;
import oroarmor.matrix.Matrix;
import oroarmor.network.NeuralNetwork;
import oroarmor.training.models.TotalError;

public class Main {

	public static void main(String[] args) {

		NeuralNetwork test = new NeuralNetwork(4);
		test.addLayer(new FeedFowardLayer(2));

		double[][] input = { { 1 }, { 2 }, { 3 }, { 4 } };

		double[][] outputs = { { 0.75 }, { 0.25 } };

		Matrix inputs = new Matrix(input);

		test.feedFoward(inputs).print();
		int total = 10000000;

		for (int i = 0; i < total; i++) {
			if (i % (total / 10) == 0) {
				test.feedFoward(inputs).print();

				test.getLayer(0).getBias().print();
			}

			test.train(inputs, new Matrix(outputs), new TotalError(0.01));
		}

		test.feedFoward(inputs).print();

	}
}
