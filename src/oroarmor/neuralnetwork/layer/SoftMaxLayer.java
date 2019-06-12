package oroarmor.neuralnetwork.layer;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.SoftMaxFunction;

public class SoftMaxLayer extends FeedFowardLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public SoftMaxLayer(int neurons) {
		super(neurons);
	}
	
	public Matrix feedFoward(Matrix inputs) {
		return super.feedFoward(inputs).applyFunction(new SoftMaxFunction());
	}

}
