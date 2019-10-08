package oroarmor.neuralnetwork.layer;

import oroarmor.neuralnetwork.matrix.Matrix;
import oroarmor.neuralnetwork.matrix.MatrixFunction;

public class RNNLayer extends FeedFowardLayer {

	public RNNLayer(int neurons) {
		super(neurons);
	}

	private static final long serialVersionUID = 1L;

	int neurons;
	int previousNeurons;
	Matrix inWeights;
	Matrix prevWeights;
	Matrix prevOut;
	

	@Override
	public Matrix feedFoward(Matrix inputs) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MatrixFunction getMatrixFunction() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getOutputNeurons() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Matrix[] getParameters() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix getWeights() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setParameters(Matrix[] parameters) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setup(int inputs) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setWeights(Matrix newWeights) {
		// TODO Auto-generated method stub

	}

}
