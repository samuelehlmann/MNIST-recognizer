package neuralNetworks;
import java.io.IOException;
import java.util.Arrays;

import fileInterface.NetworkFileReader;
import fileInterface.NetworkFileWriter;
import neuralMath.ActivationFunction;
import neuralMath.MatrixMath;
public class FullyConnectedLayer extends Layer {
	double[] biases;
	double[] biasGradientSum;

	double[][] weights;
	double[][] weightGradientSum;
	
	Layer prevLayer;
	
	public FullyConnectedLayer(int size, ActivationFunction activationFunc) {
		super(size);
		// initialize biases
		biases = new double[size];
		biasGradientSum = new double[size];
		randomizeBiases();
		
		// weights cannot be initialized until previous layer size is known
		
		this.activationFunc = activationFunc;
	}
	
	// set the previous layer and initialize weights based on its size
	@Override
	public void setPrevLayer(Layer prevLayer) {
		this.prevLayer = prevLayer;
		// if this is a new layer, randomize the weights
		if (weights == null) {
			initializeWeights(prevLayer.size);
		}
	}
	
	// initialize weights and weight gradients
	public void initializeWeights(int inputSize) {
		weights = new double[size][inputSize];
		weightGradientSum = new double[size][inputSize];
		randomizeWeights();
	}
	
	// randomize biases
	public void randomizeBiases() {
		for (int i = 0; i < size; i++) {
			biases[i] = Math.random() * 2 - 1; // random value between -1, 1
		}
	}
	
	// randomize weights
	public void randomizeWeights() {
		int columns = weights[0].length;
		
		// for each weight, make a random value
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < columns; j++) {
				weights[i][j] = Math.random() * 2 - 1; // random value between -1, 1
			}
		}
	}
	
	// forward pass of layer
	@Override
	public double[] forward(double[] input) {
		// multiply weights by input for activation
		activation = MatrixMath.multiply(weights, input);
		
		// for each activation, add biases and apply the activation function
		for (int i = 0; i < size; i++) {
			activation[i] = activation[i] + biases[i];
			activation[i] = activationFunc.activate(activation[i]);
		}
		
		return activation;
	}
	
	// reset the gradient sums to 0
	public void resetGradientSums() {
		// fill bias gradient sum with 0's
		Arrays.fill(biasGradientSum, 0);
		
		int rows = weights.length;
		
		// fill each row of weight gradient sums with 0's
		for (int i = 0; i < rows; i++) {
			Arrays.fill(weightGradientSum[i], 0);
		}
	}
	
	// learn by gradient sums
	@Override
	public void learn(double increment) {
		// learn biases and weights
		learnBiases(increment);
		learnWeights(increment);
		
		// reset gradient sums once learned
		resetGradientSums();
	}
	
	// learn biases by bias gradient sums
	public void learnBiases(double increment) {
		int rows = biases.length;
		
		// for each bias, subtract the bias gradient sum times the increment
		for (int i = 0; i < rows; i++) {
			biases[i] -= biasGradientSum[i] * increment;
		}
	}

	// learn weights by weight gradient sums
	public void learnWeights(double increment) {
		int rows = weights.length;
		int columns = weights[0].length;
		
		// for each weight, subtract the weight gradient sum times the increment
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				weights[i][j] -= weightGradientSum[i][j] * increment; 
			}
		}
	}
	
	// back propagation given the correct answer
	@Override
	public boolean backPropagate(double[] answer) {
		// calculate the bias gradient and weight gradients
		double[] biasGradient = calculateBiasGradient(answer);
		calculateWeightGradient(prevLayer.activation, biasGradient);
		// pass bias gradient and weights back
		return prevLayer.backPropagate(biasGradient, weights);
	}
	
	// back propagation given the next layer's bias gradient and weights
	@Override
	public boolean backPropagate(double[] nextBiasGradient, double[][] nextWeights) {
		// calculate the bias gradient and weights gradients
		double[] biasGradient = calculateBiasGradient(nextBiasGradient, nextWeights);
		calculateWeightGradient(prevLayer.activation, biasGradient);
		// pass bias gradient and weights back
		return prevLayer.backPropagate(biasGradient, weights);
	}
	
	// calculate bias gradient given the correct answer
	public double[] calculateBiasGradient(double[] answer) {
		int rows = biases.length;
		
		double[] biasGradient = new double[rows];
		
		// do appropriate math
		for (int i = 0; i < rows; i++) {
			biasGradient[i] = (activation[i] - answer[i]) * activationFunc.derivative(activation[i]);
			biasGradientSum[i] += biasGradient[i];
		}
		
		return biasGradient;
	}
	
	public double[] calculateBiasGradient(double[] nextGradient, double[][] nextWeights) {
		// calculate bias gradient given the next layer's bias gradient and weights
		
		// transpose next layer's weights so it can be multiplied with the next layer's gradient
		double[][] nextWeightsT = MatrixMath.transpose(nextWeights);
		
		double[] biasGradient = MatrixMath.multiply(nextWeightsT, nextGradient);
		
		int rows = biasGradient.length;
		
		
		// do appropriate math and add it to the bias gradient sum
		for (int i = 0; i < rows; i++) {
			biasGradient[i] = biasGradient[i] * activationFunc.derivative(activation[i]);
			biasGradientSum[i] += biasGradient[i];
		}
		
		return biasGradient;
	}
	
	// calculate the weight gradient given the previous layer's activation
	public double[][] calculateWeightGradient(double[] prevActivation, double[] biasGradient) {
		// transpose the previous layer's activation so it can be multiplied by the bias gradient
		double[][] prevActivationT = MatrixMath.transpose(prevActivation);
		
		double[][] weightGradient = MatrixMath.multiply(biasGradient, prevActivationT);
		
		int rows = weightGradient.length;
		int columns = weightGradient[0].length;
		
		
		// add each weight gradient to the weight gradient sum
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				weightGradientSum[i][j] += weightGradient[i][j]; 
			}
		}
		
		return weightGradient;
	}

	// write the layer to a file
	@Override
	public void writeToFile(NetworkFileWriter nfw) throws IOException {
		// write the activation function name
		nfw.writeString(activationFunc.toString());
		
		// write the size
		nfw.writeInt(size);
		
		// write the previous layer size
		nfw.writeInt(prevLayer.size);
		
		// write the biases
		nfw.writeArray(biases);
		
		// write the weights
		nfw.writeMatrix(weights);
	}

	//read a FullyConnectedLayer from a file
	static public FullyConnectedLayer readFromFile(NetworkFileReader nfr) throws IOException {
		String activationFuncType;
		int size, prevLayerSize;
		FullyConnectedLayer layer;
		
		// read the activation function type, the size, and the previous layer size
		activationFuncType = nfr.readString();
		size = nfr.readInt();
		prevLayerSize = nfr.readInt();
		
		// initialize the new layer
		layer = new FullyConnectedLayer(size, ActivationFunction.fromString(activationFuncType));
		
		// read the biases in and set the biases of the layer to those biases
		layer.biases = nfr.readArray(size);
		
		// initialize weights so the arrays are the right size 
		layer.initializeWeights(prevLayerSize);
		
		// read the weights in and set the weights of the layer to those weights
		layer.weights = nfr.readMatrix(size,  prevLayerSize);
		
		// return the loaded layer
		return layer;
	}

	// representation of the layer in a string
	@Override
	public String toString() {
		return String.format("Fully Connected Layer %d " + activationFunc, size);
	}
}
