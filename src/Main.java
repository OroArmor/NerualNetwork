import oroarmor.matrix.*;

public class Main {

	public static void main(String[] args) {

		double[][] aVals = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } };

		Matrix a = new Matrix(aVals);

		a.print();

		TanhMatrix tanh = new TanhMatrix();

		a.applyFunction(tanh).print();

		a.applyFunction(tanh).getDerivative(tanh).print();
	}

}
