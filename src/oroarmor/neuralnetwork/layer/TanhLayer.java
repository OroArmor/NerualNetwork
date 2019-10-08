package oroarmor.neuralnetwork.layer;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.MatrixFunction;
import oroarmor.neuralnetwork.matrix.SigmoidMatrix;
import oroarmor.neuralnetwork.matrix.TanhFunction;

public class TanhLayer extends FeedFowardLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public TanhLayer(int neurons) {
		super(neurons);
		// TODO Auto-generated constructor stub
	}

	@Override
	public Matrix feedFoward(Matrix inputs) {
		return weights.multiplyMatrix(inputs).applyFunction(new TanhFunction()); // tanh(W*I)
	}

	@Override
	public MatrixFunction getMatrixFunction() {
		return new TanhFunction();
	}

}
