package oroarmor.neuralnetwork.matrix;

public class SoftMaxFunction extends MatrixFunction {

	double total;

	public SoftMaxFunction() {
	}

	@Override
	public Matrix applyFunction(Matrix matrix) {
		total = matrix.collapseRows().transpose().collapseRows().getValue(0, 0);
		return matrix.divide(total);
	}

	@Override
	public Matrix getDerivative(Matrix matrix) {
		return matrix;
	}

	public double getTotal() {
		return total;
	}

}
