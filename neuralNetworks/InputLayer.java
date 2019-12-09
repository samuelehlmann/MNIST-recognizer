package neuralNetworks;
import java.io.IOException;

import fileInterface.NetworkFileReader;
import fileInterface.NetworkFileWriter;

public class InputLayer extends Layer {
	// Input Layer, most functions are dummy functions as it cannot learn and does not have biases
	
	// initialize layer
	public InputLayer(int size) {
		super(size);
		activationFunc = null;
	}
	
	// dummy setPrevLayer function
	@Override
	public void setPrevLayer(Layer prevLayer) {}

	// forward pass
	@Override
	public double[] forward(double[] input) {
		// given an input, save it and forward it
		activation = input;
		return activation;
	}
	
	// dummy backPropagate function
	@Override
	public boolean backPropagate(double[] answer) {
		return true;
	}
	
	// dummy backPropagate function
	@Override
	public boolean backPropagate(double[] nextBiasGradient, double[][] nextWeights) {
		return true;
	}
	
	// dummy learn function
	@Override
	public void learn(double increment) {}
	
	// write layer to file
	@Override
	public void writeToFile(NetworkFileWriter nfw) throws IOException {
		// write the size of the layer
		nfw.writeInt(size);
	}
	
	// create a layer from reading in a file
	static public InputLayer readFromFile(NetworkFileReader nfr) throws IOException {
		// read in the size and make a new InputLayer of that size
		int size = nfr.readInt();
		return new InputLayer(size);
	}

	// string representation of the layer
	@Override
	public String toString() {
		return String.format("Input Layer %d", size);
	}
}
