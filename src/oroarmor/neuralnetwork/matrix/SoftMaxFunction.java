package oroarmor.neuralnetwork.matrix;

public class SoftMaxFunction extends MatrixFunction {

	double total;

	public SoftMaxFunction() {
	}

	@Override
	public Matrix applyFunction(Matrix matrix) {

		matrix = matrix.exp();
		
		total = matrix.getSum();
		if(total==0) {
			return matrix;
		}
		
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
