import com.oroarmor.time.TimeFormat;
import com.oroarmor.time.TimeRemaining;

import oroarmor.layer.FeedFowardLayer;
import oroarmor.matrix.Matrix;
import oroarmor.network.NeuralNetwork;
import oroarmor.training.models.TotalError;

public class Main {

	public static void main(String[] args) {

		NeuralNetwork test = new NeuralNetwork(2);
		test.addLayer(new FeedFowardLayer(4));
		test.addLayer(new FeedFowardLayer(1));

		double[][] input1 = { { 1 }, { 1 } };
		double[][] input2 = { { 0 }, { 1 } };
		double[][] input3 = { { 1 }, { 0 } };
		double[][] input4 = { { 0 }, { 0 } };

		double[][] output1 = { { 0 } };
		double[][] output2 = { { 1 } };
		double[][] output3 = { { 1 } };
		double[][] output4 = { { 0 } };

		Matrix inputs1 = new Matrix(input1);
		Matrix inputs2 = new Matrix(input2);
		Matrix inputs3 = new Matrix(input3);
		Matrix inputs4 = new Matrix(input4);

		Matrix[] inputs = { inputs1, inputs2, inputs3, inputs4 };

		Matrix outputs1 = new Matrix(output1);
		Matrix outputs2 = new Matrix(output2);
		Matrix outputs3 = new Matrix(output3);
		Matrix outputs4 = new Matrix(output4);

		Matrix[] outputs = { outputs1, outputs2, outputs3, outputs4 };

//		int total = 10000000;

		System.out.println("Feed Foward");
		for (Matrix input : inputs) {
			test.feedFoward(input).print();
		}
		TimeRemaining timer = new TimeRemaining();
		System.out.println("Training\n");
		int trains = 1000000;
		for (int i = 0; i < trains; i++) {
			for (int j = 0; j < 4; j++) {
				test.train(inputs[j], outputs[j], new TotalError(0.075));
			}
			if(i%(trains/100) == 0) {
				System.out.println(i/(trains/100)+"%: "+TimeFormat.formatTime(timer.getTimer().getTimeElapsedSeconds() - timer.getSec((double) i/(trains))));
			}
		}
		System.out.println("Feed Foward after training");
		
		double error = 0;
		for (int i = 0; i < 4; i++) {
			error +=Math.abs( outputs[i].subtractMatrix(test.feedFoward(inputs[i]).print()).getValue(0,0));
		}
		System.out.println("Error: "+error);
	}
}
