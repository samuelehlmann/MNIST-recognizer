package neuralNetworks;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class NeuralNetwork {
	// neural network object
	
	// list of layers and the number of layers
	public ArrayList<Layer> layers;
	public int numLayers;
	
	// constructor
	public NeuralNetwork() {
		// initialize list of layers and numLayers
		layers = new ArrayList<Layer>();
		numLayers = 0;
	}
	
	// add a layer to the network
	public void addLayer(Layer L) {
		// add the layer to the list and increase the number of layers
		layers.add(L);
		numLayers++;
		// if this is not the first layer, give the layer its previous layer
		if (numLayers != 1) {
			layers.get(numLayers-1).setPrevLayer(layers.get(numLayers-2));
		}
	}
	
	
	// forward pass through the network
	public double[] forward(Digit digit) {
		// get input
		double[] intermediate = digit.input;
		
		// for each layer, pass the returned array to the next
		for (int i = 0; i < numLayers; i++) {
			intermediate = layers.get(i).forward(intermediate);
		}
		
		// return the value
		return intermediate;
	}
	
	// back propagation
	public void backPropagate(Digit digit) {
		// turn the answer into a one-hot vector
		double[] answer = new double[10];
		answer[digit.answer]= 1; 
		
		// pass the answer to the last layer, it will pass it along from there
		layers.get(numLayers-1).backPropagate(answer);
	}
	
	// learn by an increment based on gradient sums
	public void learn(double increment) {
		// tell each layer to learn based on that increment
		for (int i = 0; i < numLayers; i++) {
			layers.get(i).learn(increment);
		}
	}
	
	// train the neural network
	public void train(int batchSize, int maxEpochs, int minAccuracy, double learningRate, Digit[] trainingData) { 
		// if maxEpochs or minAccuracy is negative they will be ignored
		// minAccuracy should be a percentage (e.g 90, not 0.9)
		int numExamples = trainingData.length;
		double increment = learningRate / batchSize;
		double[] result;
		
		// for each epoch
		for (int i = 0; i != maxEpochs; i++) { // if maxEpochs is negative, it will never be reached
			// vectors to keep track of number correct for each and total of each digit 0-9
			int[] numCorrect = new int[10];
			int[] numTotal = new int[10];
		
			System.out.printf("Beginning epoch %d%n", i);
			
			// randomize order of digits
			Collections.shuffle(Arrays.asList(trainingData));
			
			// for each batch
			for (int j = 0; j <= numExamples - batchSize; j += batchSize) {
				// for each digit in the batch
				for (int k = 0; k < batchSize; k++) {
					Digit digit = trainingData[j+k];
					
					// forward the digit through the network and get the numerical answer
					result = forward(digit);
					int answer = convertToAnswer(result);
					
					// add the correct digit to the total number of each digit, and if the network was correct, add it to the number correct, too
					numTotal[digit.answer] += 1;
					if (digit.answer == answer) {
						numCorrect[answer] += 1;
					}
					
					// back propagate the digit
					backPropagate(digit);
				}
				// at the end of the batch, learn
				learn(increment);
			}
			
			// calculate the total number of examples and total correct numbers of examples
			int numCorrectTotal = 0;
			int numTotalTotal = 0;
			
			// for each digit 0-9, add its numbers to the totals and print its statistics
			for (int l = 0; l < 10; l++) {
				numCorrectTotal += numCorrect[l];
				numTotalTotal += numTotal[l];
				
				// print digits' statistics
				System.out.printf("%d = %4d/%4d ", l, numCorrect[l], numTotal[l]);
				if (l == 5) { // line break after first five digits
					System.out.println();
				}
			}
			
			// calculate total accuracy and print it out
			double accuracy = (double) numCorrectTotal / numTotalTotal * 100;
			System.out.printf("Accuracy = %d/%d = %6.3f%%%n%n", numCorrectTotal, numTotalTotal, accuracy);
			
			// if minimum accuracy has been reached, break, unless accuracy is negative
			if (accuracy > minAccuracy && minAccuracy >= 0) {
				break;
			}
		}
	}
	
	// test the network on a data set
	public void test(Digit[] testData) {
		int numData = testData.length;
		
		// vectors to keep track of number correct for each and total of each digit 0-9
		int[] numCorrect = new int[10];
		int[] numTotal = new int[10];
		double[] result;
		
		System.out.printf("Beginning testing%n");
		
		// for each Digit, forward it through the network and add it to the statistics
		for (int i = 0; i < numData; i++) {
			Digit digit = testData[i];
			
			// forward the digit and convert its answer to an integer
			result = forward(digit);
			int answer = convertToAnswer(result);
			
			// add the correct digit to the total number of each digit, and if the network was correct, add it to the number correct, too 
			numTotal[digit.answer] += 1;
			if (digit.answer == answer) {
				numCorrect[answer] += 1;
			}
		}
		
		// calculate the total number of examples and total correct numbers of examples
		int numCorrectTotal = 0;
		int numTotalTotal = 0;
		
		// for each digit 0-9, add its numbers to the totals and print its statistics
		for (int l = 0; l < 10; l++) {
			numCorrectTotal += numCorrect[l];
			numTotalTotal += numTotal[l];
			
			// print digits' statistics
			System.out.printf("%d = %4d/%4d ", l, numCorrect[l], numTotal[l]);
			if (l == 5) { // line break after first five digits
				System.out.println();
			}
		}
		
		// calculate total accuracy and print it out
		double accuracy = (double) numCorrectTotal / numTotalTotal * 100;
		System.out.printf("Accuracy = %d/%d = %6.3f%%%n%n", numCorrectTotal, numTotalTotal, accuracy);
	}
	
	
	// convert a result vector to an integer answer (classification)
	public int convertToAnswer(double[] result) {
		int answer = 0;
		
		// for each value in the vector, save its if it is larger than that of the previously saved index
		for (int j = 1; j < 10; j++) {
			answer = (result[j] > result[answer]) ? j : answer;
		}
		
		return answer;
	}
	
	// return a string representation of the network
	public String toString() {
		String str = "";
		
		// add each layer's string representation to the string
		for (int i = 0; i < numLayers; i++) {
			str += layers.get(i);
			
			// if it is not the last layer, add a line break before the next layer
			if (i < numLayers - 1) {
				str += "%n";
			}
		}
		return str;
	}
}
