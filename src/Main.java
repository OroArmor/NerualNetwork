import oroarmor.matrix.Matrix;

public class Main {

	public static void main(String[] args) {

		double[][] aVals = { { 1 , 2 ,  3 }, 
							 { 4 , 5 ,  6 }, 
							 { 7 , 8 ,  9 }, 
							 { 10, 11, 12 } 
						};
		double[][] bVals = { { 1,  2 , 3 , 4 , 5 , 6  }, 
							 { 7 , 8 , 9 , 10, 11, 12 }, 
							 { 13, 14, 15, 16, 17, 18 } 
						};

		Matrix a = new Matrix(aVals);
		Matrix b = new Matrix(bVals);

		a.multiplyMatrix(b).divide(1000).print();
	}

}
