package oroarmor.neuralnetwork.matrix;

public class SoftMaxFunction extends MatrixFunction {

	public SoftMaxFunction() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public Matrix applyFunction(Matrix matrix) {
		return matrix.divide(matrix.collapseRows().transpose().collapseRows().getValue(0, 0));
	}

	@Override
	public Matrix getDerivative(Matrix matrix) {
		return matrix;
	}

}
