package neuralMath;

import java.util.Arrays;

public class NDDArray {
	protected int[] size;
	protected int length;
	protected int[] multiplicationTable;
	protected int[] transpositionTable;
	protected double[] array;
	
	public NDDArray(int...size) {
		this.length = 1;
		
		this.size = size;
		transpositionTable = new int[size.length];
		
		for (int i=0; i<size.length; i++) {
			this.length *= size[i];
			transpositionTable[i] = i;
		}
		
		calculateMultiplicationTable();
		
		this.array = new double[this.length];
	}
	
	public NDDArray(double[] array, int...size) {
		this.length = 1;
		
		this.size = size;
		transpositionTable = new int[size.length];
		
		for (int i=0; i<size.length; i++) {
			this.length *= size[i];
			transpositionTable[i] = i;
		}
		
		calculateMultiplicationTable();
		
		this.array = array;
	}
	
	private void calculateMultiplicationTable() {
		multiplicationTable = new int[size.length];
		Arrays.fill(multiplicationTable, 1);
		
		for (int i=0; i<size.length; i++) {
			for (int j=0; j<i; j++) {
				multiplicationTable[j] *= size[i];
			}
		}
	}
	
	public double get(int...indices) {
		return array[calculateIndex(indices)];
	}
	
	public void set(double value, int...indices) {
		array[calculateIndex(indices)] = value;
	}
	
	public int calculateIndex(int...indices) {
		int index = 0;
		for (int i=0; i<indices.length; i++) {
			index += indices[i] * multiplicationTable[transpositionTable[i]];
		}
		return index;
	}
	
	public void resize(int...newSize) {
		int newLength = 1;
		
		for (int i=0; i<newSize.length; i++) {
			newLength *= newSize[i];
		}
		
		if (newLength == length) {
			size = newSize;
			calculateMultiplicationTable();
		}
		else {
			System.out.printf("Error: resize indices don't match length. Nothing will happen.%n");
		}
	}
	
	public void transpose(int dim1, int dim2) {
		int index = transpositionTable[dim1];
		transpositionTable[dim1] = transpositionTable[dim2];
		transpositionTable[dim2] = index;
	}
	
	public int[] size() {
		return size;
	}
	
	public int length() {
		return length;
	}
}
