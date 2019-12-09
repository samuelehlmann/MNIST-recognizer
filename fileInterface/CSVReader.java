package fileInterface;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import neuralNetworks.Digit;

public class CSVReader {
	// reader for the MNIST data
	public static Digit[] readData(String file, int size) throws IOException {
		// create array of given size
		Digit[] data = new Digit[size];
		
		BufferedReader br = new BufferedReader(new FileReader(file));
		
		String str;
		
		// read each value in
		for (int i = 0; i < size; i++) {
			str = br.readLine();
			
			// answer is first value
			int answer = Integer.parseInt(str.substring(0,1));
			
			// image is all subsequent values
			String[] strValues = str.substring(2).split(",");
			double[] numValues = new double[784];
			
			// scale each value to between 0 and 1
			for (int j = 0; j < 784; j++) {
				numValues[j] = ((double) Integer.parseInt(strValues[j])) / 255.0;
			}
			// insert it into the array
			data[i] = new Digit(answer, numValues);
		}

		br.close();
		
		// return the data in Digit format
		return data;
	}
}
