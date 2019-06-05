import oroarmor.layer.FeedFowardLayer;
import oroarmor.matrix.Matrix;
import oroarmor.network.NeuralNetwork;
import oroarmor.training.models.TotalError;

public class Main {

	public static void main(String[] args) {

		NeuralNetwork test = new NeuralNetwork(2);
		test.addLayer(new FeedFowardLayer(4));
		test.addLayer(new FeedFowardLayer(4));
		test.addLayer(new FeedFowardLayer(1));

		double[][] input = { { 1, 1 } };

		double[][] output = { { 0 } };

		Matrix inputs = new Matrix(input);

//		int total = 10000000;

		System.out.println("Feed Foward");
		test.feedFoward(inputs).print();

		
		System.out.println("Training\n");
		test.train(inputs, new Matrix(output), new TotalError(0.01));

		System.out.println("Feed Foward after training");
		test.feedFoward(inputs).print();
		
//		for (int i = 0; i < total; i++) {
//			if (i % (total / 10) == 0) {
//				test.feedFoward(inputs).print();
//			}
//			test.train(inputs, new Matrix(outputs), new TotalError(0.01));
//		}
//		test.feedFoward(inputs).print();
	}
}
