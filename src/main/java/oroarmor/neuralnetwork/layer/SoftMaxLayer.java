package oroarmor.neuralnetwork.layer;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.Matrix.MatrixType;
import oroarmor.neuralnetwork.matrix.function.SoftMaxFunction;

public class SoftMaxLayer<T extends Matrix<T>> extends FeedFowardLayer<T> {

	/**
	 *
	 */
	private static final long serialVersionUID = 13L;

	SoftMaxFunction softMax = new SoftMaxFunction();

	public SoftMaxLayer(int neurons, MatrixType type) {
		super(neurons, type);
	}

	@Override
	public T feedFoward(T inputs) {
		return super.feedFoward(inputs).applyFunction(softMax);
	}

	public SoftMaxFunction getSoftMaxFunction() {
		return softMax;
	}

}
