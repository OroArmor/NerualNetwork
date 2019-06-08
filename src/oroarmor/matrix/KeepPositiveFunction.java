package oroarmor.matrix;

public class KeepPositiveFunction extends MatrixFunction {

	@Override
	public Matrix applyFunction(Matrix matrix) {
		Matrix newMatrix = new Matrix(matrix.getRows(), matrix.getCols());

		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				newMatrix.setValue(i, j, keepPos(matrix.getValue(i, j)));
			}
		}

		return newMatrix;
	}
	
	private double keepPos(double val) {
		if(val > 0) {
			return  val;
		}
		return 0;
	}

	private double dKeepPos(double val) {
		if(val > 0) {
			return  1;
		}
		return 0;
	}
	
	@Override
	public Matrix getDerivative(Matrix matrix) {
		Matrix newMatrix = new Matrix(matrix.getRows(), matrix.getCols());

		for (int i = 0; i < matrix.getRows(); i++) {
			for (int j = 0; j < matrix.getCols(); j++) {
				newMatrix.setValue(i, j, dKeepPos(matrix.getValue(i, j)));
			}
		}

		return newMatrix;
	}

}
