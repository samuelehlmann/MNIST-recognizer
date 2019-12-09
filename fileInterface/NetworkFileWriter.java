package fileInterface;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import neuralNetworks.Layer;
import neuralNetworks.NeuralNetwork;

public class NetworkFileWriter {
	// object to write neural networks to files
	BufferedWriter bw;
	
	public NetworkFileWriter(File file) throws IOException {
		// create a network file writer and initialize its buffered writer
		bw = new BufferedWriter(new FileWriter(file));
	}
	
	public static void saveNetwork(NeuralNetwork network, File file) throws IOException {
		// save a network to a file
		
		// get the number of layers and initialize the writer
		int numLayers = network.layers.size();
		NetworkFileWriter nfw = new NetworkFileWriter(file);
		
		// write the number of layers then write each layer
		nfw.writeInt(numLayers);
		for (int i = 0; i < numLayers; i++) {
			nfw.writeLayer(network.layers.get(i));
		}
		
		// close the writer
		nfw.close();
	}
	
	public void writeLayer(Layer layer) throws IOException {
		// write a layer to a file
		// write the layer name and then extra format for clarity
		bw.write(layer.getClass().getName() + "\n{\n");
		// let the layer itself write its data to the file
		layer.writeToFile(this);
		// close the brackets
		bw.write("}\n");
	}

	public void writeInt(int i) throws IOException {
		// write an integer to the file
		bw.write(i + "\n");
	}
	
	public void writeArray(double[] numValues) throws IOException {
		// write an array to the file
		
		bw.write("{\n"); // bracket for clarity
		
		int size = numValues.length;
		
		// write each value of the array separated by commas
		for (int j = 0; j < size-1; j++) {
			bw.write(numValues[j] + ",");
		}
		bw.write(numValues[size-1] + "\n");
		
		bw.write("}\n"); // bracket for clarity
	}
	
	public void writeMatrix(double[][] numValues) throws IOException {
		// write a matrix to a file
		bw.write("{\n"); // bracket for clarity
		
		int rows = numValues.length;
		int columns = numValues[0].length;
		
		// write each value, separated by commas, with rows separated by line breaks
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns-1; j++) {
				bw.write(numValues[i][j] + ",");
			}
			// final value does not have comma but line break instead
			bw.write(numValues[i][columns-1] + "\n");
		}
		
		bw.write("}\n"); // bracket for clarity
	}

	public void writeString(String str) throws IOException {
		// write a string to the file
		bw.write(str + "\n");;
	}
	
	public void close() throws IOException {
		// close the writer
		bw.close();
	}
}
