package fileInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import neuralNetworks.FullyConnectedLayer;
import neuralNetworks.InputLayer;
import neuralNetworks.Layer;
import neuralNetworks.NeuralNetwork;

public class NetworkFileReader {
	// reader for reading networks from files
	BufferedReader br;
	
	public NetworkFileReader(File file) throws FileNotFoundException {
		// create a file reader
		br = new BufferedReader(new FileReader(file));
	}
	
	public static NeuralNetwork loadNetwork(File file) throws IOException {
		// function to load a neural network from a given file
		// initialize a neural network and a new network file reader
		NeuralNetwork network = new NeuralNetwork();
		NetworkFileReader nfw = new NetworkFileReader(file);
		
		// read in number of layers
		int numLayers = nfw.readInt();
		
		Layer layer;
		
		// for the number of layers, read in a layer and add it to the network
		for (int i = 0; i < numLayers; i++) {
			layer = nfw.readLayer();
			network.addLayer(layer);
		}
		
		// close the network file reader and return the network		
		nfw.close();
		return network;
	}
	
	public Layer readLayer() throws IOException {
		// read in a layer of the network
		// read in the type of the network
		String str;
		str = br.readLine();
		
		br.readLine(); // read in "{"
		
		Layer layer = null;
		
		// depending on the type of layer, give that given class the network file reader to read in the layer
		switch (str) {
		case "neuralNetworks.InputLayer":
			layer = InputLayer.readFromFile(this);
			break;
		case "neuralNetworks.FullyConnectedLayer":
			layer = FullyConnectedLayer.readFromFile(this);
			break;
		}
		
		br.readLine(); // read in "}"
		
		// return the generated layer
		return layer;
	}

	public int readInt() throws IOException {
		// read in an integer
		String str = br.readLine();
		return Integer.parseInt(str);
	}
	
	public double[] readArray(int size) throws IOException {
		// read in an array
		
		br.readLine(); // read in "{"
		
		String str = br.readLine();
		
		// split the line into values
		String[] strValues = str.split(",");
		double[] numValues = new double[size];
		
		// convert the values to doubles and add them to the output array
		for (int j = 0; j < size; j++) {
			numValues[j] = Double.parseDouble(strValues[j]);
		}
		
		br.readLine(); // read in "}"
		
		// return the array
		return numValues;
	}
	
	public double[][] readMatrix(int rows, int columns) throws IOException {
		// read in a matrix
		br.readLine(); // read in "{"
		
		double[][] numValues = new double[rows][columns];
		
		// for each row of the matrix, read a new line and split it up
		for (int i = 0; i < rows; i++) {
			String str = br.readLine();
			String[] strValues = str.split(",");
			
			// convert each value to a double and add it to the matrix
			for (int j = 0; j < columns; j++) {
				numValues[i][j] = Double.parseDouble(strValues[j]);
			}
		}
		
		br.readLine(); // read in "}"
		
		// return the matrix
		return numValues;
	}

	public String readString() throws IOException {
		// read in a string
		return br.readLine();
	}
	
	public void close() throws IOException {
		// close the reader
		br.close();
	}
}
