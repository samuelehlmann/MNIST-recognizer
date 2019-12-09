package neuralNetworks;

public class Digit {
	// data type to store a digit's representation and its answer
	public int answer; // answer stored as int, converted to one-hot vector in training/testing
	public double[] input = new double[784];
	
	public Digit(int answer, double[] input) {
		this.answer = answer;
		this.input = input;
	}
}
