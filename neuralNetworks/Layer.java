package neuralNetworks;
import java.io.IOException;

import fileInterface.NetworkFileWriter;
import neuralMath.ActivationFunction;

public abstract class Layer {
	// template for layers
	public int size;
	double[] activation;
	public ActivationFunction activationFunc;
	
	public Layer(int size) {
		this.size = size;
		activation = new double[size];
	}

	// methods layer types must implement
	abstract void setPrevLayer(Layer prevLayer);
	abstract double[] forward(double[] input);
	abstract boolean backPropagate(double[] answer);
	abstract boolean backPropagate(double[] nextBiasGradient, double[][] nextWeights);
	abstract void learn(double increment);
	public abstract void writeToFile(NetworkFileWriter nfw) throws IOException;
	public abstract String toString();
}
