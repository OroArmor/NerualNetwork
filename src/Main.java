import oroarmor.layer.FeedFowardLayer;
import oroarmor.matrix.Matrix;
import oroarmor.network.NeuralNetwork;
import oroarmor.training.models.TotalError;

public class Main {

	public static void main(String[] args) {

		NeuralNetwork test = new NeuralNetwork(4);
		test.addLayer(new FeedFowardLayer(1));
		
		double[][] inputs = {{1},{2},{3},{4}};
		
		test.feedFoward(new Matrix(inputs)).print();
		
		double[][] outputs = {{1}};
		
		test.train(new Matrix(inputs), new Matrix(outputs), new TotalError(0.1));
		
//		Matrix a = new Matrix(inputs);
//		
//		a.print();
//		a.transpose().print();
		
	}
}
