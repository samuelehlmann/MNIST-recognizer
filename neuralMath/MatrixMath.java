package neuralMath;

public final class MatrixMath { 
	// helpful methods to do matrix math
	private MatrixMath() {}
	
	// multiply two matrices
	public static double[][] multiply(double[][] m1, double[][] m2) {
		int rows = m1.length;
		int middle = m2.length;
		int columns = m2[0].length;
		
		// new matrix of appropriate size
		double[][] m3 = new double[rows][columns];
		
		// matrix multiply, summing values into appropriate value of new matrix
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++)  {
				m3[i][j] = 0;
				for (int k = 0; k < middle; k++) {
					m3[i][j] += m1[i][k] * m2[k][j];
				}
			}
		}
		
		// return matrix
		return m3;
	}
	
	// multiply matrix and single column matrix (vector)
	public static double[] multiply(double[][] m1, double[] m2) {
		// implied
		int rows = m1.length;
		int middle = m2.length;
		// columns implied 1
		
		// initialize new single column vector (matrix) of correct size)
		double[] m3 = new double[rows];
		
		// multiply
		for (int i = 0; i < rows; i++) {
			m3[i] = 0;
			for (int k = 0; k < middle; k++) {
				m3[i] += m1[i][k] * m2[k];
			}
		}
		
		return m3;
	}
	
	// multiply single column matrix (vector) with single row matrix
	public static double[][] multiply(double[] m1, double[][] m2) {
		int rows = m1.length;
		// middle implied 1
		int columns = m2[0].length;
		
		// initialize new matrix of correct size
		double[][] m3 = new double[rows][columns];
		
		// multiply
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				m3[i][j] = m1[i] * m2[0][j];
			}
		}
		
		return m3;
	}
	
	// transpose a matrix
	public static double[][] transpose(double[][] m1) {
		int rows = m1.length;
		int columns = m1[0].length;
		
		// new matrix of appropriate size
		double[][] m2 = new double[columns][rows];
		
		// switch rows, columns
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++)  {
				m2[j][i] = m1[i][j];
			}
		}
		
		return m2;
	}
	
	// transpose a single column matrix (vector) (into a single row matrix)
	public static double[][] transpose(double[] m1) {
		int rows = m1.length;
		
		// new matrix
		double[][] m2 = new double[1][rows];
		
		// switch rows, columns
		for (int i = 0; i < rows; i++) {
			m2[0][i] = m1[i];
		}
		
		return m2;
	}
}
