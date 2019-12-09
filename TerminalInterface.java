/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MNIST Digit Recognition Software
// by Samuel Ehlmann
// CWID 102-49-611
// Assignment 2 - CSC 475
// 
// This program allows the user to generate, train, test, save, and load neural networks for the MNIST data set to recognize digits
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import neuralNetworks.*;

import java.io.*;
import java.util.function.Supplier;

import fileInterface.*;
import neuralMath.ActivationFunction;

public class TerminalInterface {
	// constants
	static final String TRAININGFILE = "mnist_train.csv";
	static final String TESTINGFILE = "mnist_test.csv";
	static final File SAVEFILEFOLDER = new File("Saved_Networks");	
	static final char[] ASCIILIST = " .'^\",:;li~+-?1tfrxczXYULQZ#W&%@".toCharArray();
	static boolean DEBUG = false;
	
	// variables
	static Digit[] trainingSet;
	static Digit[] testingSet;
	
	static NeuralNetwork nn;
	
	static boolean networkTrained;
	
	static BufferedReader br;
	
	public static void main(String[] args) throws IOException {
		// initialize buffered reader for user input
		br = new BufferedReader(new InputStreamReader(System.in));
		
		// network is initially not trained
		networkTrained = false;
		
		// load training and testing sets
		trainingSet = CSVReader.readData(TRAININGFILE, 60000);
		testingSet = CSVReader.readData(TESTINGFILE, 10000);
		
		// initialize neural network as default
		nn = defaultNeuralNetwork();
		
		// loop until the user selects to exit, displaying the appropriate commands for a trained/untrained network
		while (true) {
			if (networkTrained) {
				if (commandMenu(trainedCommands, "Menu (trained network):", "Exit") == -1) break;
			}
			else {
				if (commandMenu(untrainedCommands, "Menu (untrained network):", "Exit") == -1) break;
			}
			
		}
		
		br.close();
	}
	
	static NeuralNetwork defaultNeuralNetwork() {
		// set up default network and return it
		NeuralNetwork network = new NeuralNetwork();
		network.addLayer(new InputLayer(784));
		network.addLayer(new FullyConnectedLayer(15, ActivationFunction.sigmoid));
		network.addLayer(new FullyConnectedLayer(10, ActivationFunction.sigmoid));
		return network;
	}
	
	static int commandMenu(Command[] cmds, String menuTitle, String exitOption) {
		// returns index of command chosen, or -1 if exit option chosen
		clearScreen();
		// repeat until a valid value is chosen
		while (true) {
			// print menu title, a number associated with each command and the command description, and the exit option with a prompt to enter a number
			System.out.printf(menuTitle + "%n");
			for (int i = 0; i < cmds.length; i++) {
				System.out.printf("[%d] " + cmds[i] + "%n", i + 1);
			}
			System.out.printf("%n[0] " + exitOption + " %n%n Type a number and press enter: ");

			// read in values from the user and if they are within the range of options, execute that command and return the index of the selection
			try {
				String str = br.readLine();
				int response = Integer.parseInt(str);
				if (response > 0 && response <= cmds.length) {
					cmds[response-1].execute();
					return response-1;
				}
				else if (response == 0) {
					return response-1;
				}
			}
			catch (IOException e) {
				if (DEBUG) e.printStackTrace();
			}
			catch (NumberFormatException e) {
				if (DEBUG) e.printStackTrace();
			}
			// if a value wasn't returned, the user did not input a valid choice
			clearScreen();
			System.out.printf("Invalid input. Please enter one of the numbers in the menu.%n%n");			
		}
	}
	
	static class Command {
		// Command class for use in command menus, to wrap functions to create menu entries
		String description; // description for the user to see
		Supplier<Boolean> function; // function the command calls
		
		public Command(Supplier<Boolean> function, String description) {
			// initialize command
			this.function = function;
			this.description = description;
		}
		
		public boolean execute() {
			// execute the function
			return function.get();
		}
		
		public String toString() {
			// return the description
			return description;
		}
	}
	
	static final Command defaultTrainCmd = new Command(TerminalInterface::defaultTrain, "Train the network");	
	static boolean defaultTrain() {
		// train using default options
		if (networkTrained) {
			// warn user the network is already trained, and let user decide to replace it or keep it, or cancel
			int response = optionsMenu("There is already a trained network.", new String[] {"Replace the current network", "Keep training the current network"});
			switch (response) {
			case -1: // cancel option
				return false;
			case 0: // replace option
				nn = defaultNeuralNetwork();
				break;
			case 1: // keep option
				break;
			}
		}
		
		// train the network and update networkTrained
		clearScreen();
		nn.train(10, 30, -1, 3, trainingSet);
		networkTrained = true;
		enterToReturn();
		return true;
	}
	
	static final Command loadCmd = new Command(TerminalInterface::load, "Load a pre-trained network");
	static boolean load() {
		// load a network from a file
		clearScreen();
		
		if (!SAVEFILEFOLDER.exists()) SAVEFILEFOLDER.mkdir();
		
		// get possible saved files
		String[] files = SAVEFILEFOLDER.list();
		
		// check that there are files
		if (files.length == 0) {
			System.out.printf("There are no saved files.%n");
			enterToReturn();
			return false;
		}
		
		// let user select a file
		int response = optionsMenu("Select a file to load:", files);
		
		// if cancel response, stop
		if (response == -1) return false; 
		
		// construct file name
		String saveFile = SAVEFILEFOLDER.getPath() + "/" + files[response];
		
		clearScreen();
		
		if (networkTrained) {
			// warn the user there is an already trained network, and ask if they would like to replace it
			response = optionsMenu("Would you like to replace the currently loaded network with this network?", new String[] {"Yes, replace it"});
			switch (response) {
			case -1: // cancel option
				return false;
			case 0: // keep option
				break;
			}
		}
		// try to load the network from the file
		try {
			nn = NetworkFileReader.loadNetwork(new File(saveFile));
		} 
		catch (FileNotFoundException e) { // if there is no file, let the user know
			if (DEBUG) e.printStackTrace();
			System.out.printf("There is no saved network by that file name.%n");
			enterToReturn();
			return false;
		} //
		catch (IOException e) { // if there is another error, let the user know
			if (DEBUG) e.printStackTrace();
			System.out.printf("There was an error loading the file.%n");
			enterToReturn();
			return false;
		}
		
		// update networkTrained value and let user know it was loaded
		networkTrained = true;
		
		System.out.printf("Network successfully loaded.%n");
		enterToReturn();
		
		return true;
	}
	
	static final Command displayTrainingAccuracyCmd = new Command(TerminalInterface::displayTrainingAccuracy, "Display network accuracy on training data");
	static boolean displayTrainingAccuracy() {
		// display the accuracy on the training data
		clearScreen();
		System.out.printf("Displaying accuracy on training data:%n");
		displayAccuracy(trainingSet);
		return true;
	}
	
	static final Command displayTestingAccuracyCmd = new Command(TerminalInterface::displayTestingAccuracy, "Display network accuracy on testing data");
	static boolean displayTestingAccuracy() {
		// display the accuracy on the testing data
		clearScreen();
		System.out.printf("Displaying accuracy on testing data:%n");
		displayAccuracy(testingSet);
		return true;
	}
	
	static boolean displayAccuracy(Digit[] testingData) {
		// display the accuracy on a given data set
		nn.test(testingData);
		enterToReturn();
		return true;
	}
	
	static final Command saveNetworkCmd = new Command(TerminalInterface::saveNetwork, "Save the network state to file");
	static boolean saveNetwork() {
		String saveFile = "";
		
		if (!SAVEFILEFOLDER.exists()) SAVEFILEFOLDER.mkdir();
		
		clearScreen();
		
		// ask for save file name
		System.out.printf("%n Enter a file name: ");
		try {
			saveFile = SAVEFILEFOLDER.getPath() + "/" + br.readLine();
		} catch (IOException e1) {
			if (DEBUG) e1.printStackTrace();
		}
		
		clearScreen();
		
		// try to save the network to a file
		try {
			NetworkFileWriter.saveNetwork(nn, new File(saveFile));
			System.out.printf("Network successfully saved.%n");
			enterToReturn();
			return true;
		} 
		catch (IOException e) { // if there was an error saving the file, let the user know
			if (DEBUG) e.printStackTrace();
			System.out.printf("Error saving network.%n");
			enterToReturn();
			return false;
		}
	}
	
	static final Command displayImagesCmd = new Command(TerminalInterface::displayImages, "Display image classification on testing data");
	static boolean displayImages() {
		// display images (with correct images)
		testImages(testingSet, false);
		return true;
	}
	
	static final Command displayIncorrectImagesCmd = new Command(TerminalInterface::displayIncorrectImages, "Display incorrect image classification on testing data");
	static boolean displayIncorrectImages() {
		// display images without correct images
		testImages(testingSet, true);
		return true;
	}
	
	static boolean testImages(Digit[] testData, boolean skipCorrect) {
		// display images
		int numData = testData.length;
		double[] result;
		
		// go through testData
		for (int i = 0; i < numData; i++) {
			// get an integer answer for given digit to be recognized
			Digit digit = testData[i];
			result = nn.forward(digit);
			int answer = nn.convertToAnswer(result);
			
			if (!skipCorrect) { // if not skipping correct, display all images
				displayImage(digit, i, answer);
				if (!continueOrQuit()) { // if user does not decide to continue, quit
					break;
				}
			}
			else if (digit.answer != answer) { // otherwise, only display images where actual answer and neural network answer do not match
				displayImage(digit, i, answer);
				if (!continueOrQuit()) { // if user does not decide to continue, quit
					break;
				}
			}
		}
		return true;
	}
	
	static void displayImage(Digit image, int testNumber, int networkAnswer) {
		// display an individual digit
		// for each value in the 28x28 image (array of size 784), print the corresponding ascii character
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				// scale image input back up by 255
				// divide by 8 as ASCIILIST has 32 characters (256/32=8)
				System.out.print(ASCIILIST[((int) (image.input[i*28+j]*255))/8]);
			}
			System.out.printf("%n");
		}
		
		// display info on the classification
		System.out.printf("%nTesting case #%d:  Correct classification: %d  Network Output: %d  ", testNumber, image.answer, networkAnswer);
		System.out.printf(networkAnswer == image.answer ? "Correct.%n" : "Incorrect.%n");
	}
	
	static final int optionsMenu(String prompt, Object[] options) {
		// returns the index of the option selected
		// used to create options that don't have any function associated
		// populate a list of commands with dummy functions that always return true
		Command[] cmds = new Command[options.length];
		for (int i = 0; i < options.length; i++) {
			cmds[i] = new Command(() -> true, options[i].toString());
		}
		// return the index that the user selects from a command menu
		return commandMenu(cmds, prompt, "Cancel");
	}
	
	static final Command customNetworkCmd = new Command(TerminalInterface::customNetwork, "Create a custom network configuration");
	static boolean customNetwork() {
		// let the user design a custom network
		int response;
		
		if (networkTrained) {
			// verify the user would like to replace the current, already trained network
			response = optionsMenu("Would you like to replace the currently loaded network?", new String[] {"Yes, replace it"});
			switch (response) {
			case -1: // cancel option
				return false;
			case 0: // replace option
				break;
			}
		}
		
		// initialize new neural network
		NeuralNetwork network = new NeuralNetwork();
		int numHiddenLayers;
		
		// let user select number of hidden layers
		while (true) {
			System.out.printf("%nNumber of hidden Layers: ");
			// make sure the user inputs a non-negative integer, or tell them it is invalid input and repeat
			try {
				String str = br.readLine();
				response = Integer.parseInt(str);
				if (response >= 0) {
					numHiddenLayers = response;
					break;
				}
			}
			catch (IOException e) {
				if (DEBUG) e.printStackTrace();
			}
			catch (NumberFormatException e) {
				if (DEBUG) e.printStackTrace();
			}
			System.out.printf("%nInvalid input. Please input a integer greater than or equal to zero.");
		}
		
		// add input layer
		network.addLayer(new InputLayer(784));
		
		// for each hidden layer
		for (int i = 0; i < numHiddenLayers; i++) {
			int nodes;
			// let user select number of nodes in hidden layer
			while (true) {
				System.out.printf("%nNumber of nodes in hidden layer %d: ", i);
				// make sure the user inputs a positive integer, or tell them it is invalid input and repeat
				try {
					String str = br.readLine();
					response = Integer.parseInt(str);
					if (response > 0) {
						nodes = response;
						break;
					}
				}
				catch (IOException e) {
					e.printStackTrace();
				}
				catch (NumberFormatException e) {
					
				}
				System.out.printf("%nInvalid input. Please input a positive integer.");
			}
			
			// let user select activation function
			response = optionsMenu(String.format("Activation function for hidden layer %d:", i), ActivationFunction.list);
			if (response == -1) break; // if user canceled, break
			
			ActivationFunction af = ActivationFunction.list[response];
			
			network.addLayer(new FullyConnectedLayer(nodes, af));
		}
		// get activation function for final layer, if user has not canceled
		if (response != -1) {
			response = optionsMenu("Activation function for final layer (sigmoid greatly recommended):", ActivationFunction.list);
		}
		
		// if user has canceled at any point, acknowledge that and exit
		if (response == -1) {
			clearScreen();
			System.out.printf("Custom neural network creation canceled.%n");
			enterToReturn();
			return false;
		}
		
		// if user hasn't canceled, continue
		ActivationFunction af = ActivationFunction.list[response];
		
		network.addLayer(new FullyConnectedLayer(10, af)); // add final layer
		
		// replace current neural network with new one
		nn = network;
		
		networkTrained = false;
		
		return true;
	}
	
	static final Command customTrainCmd = new Command(TerminalInterface::customTrain, "Train the network on custom parameters");
	static boolean customTrain() {
		// let the user select the training parameters
		if (networkTrained) { 
			// warn user the network is already trained, and let user decide to replace it or keep it, or cancel
			int response = optionsMenu("There is already a trained network.", new String[] {"Replace the current network", "Keep training the current network"});
			switch (response) {
			case -1: // cancel option
				return false;
			case 0: // replace option
				nn = defaultNeuralNetwork();
				break;
			case 1: // cancel option
				break;
			}
		}
		// initialize training parameter variables
		int batchSize;
		int epochs;
		float learningRate;

		clearScreen();
		
		// let user input batch size
		while (true) {
			// ensure batch size is a positive integer, or tell the user the input is invalid and repeat
			System.out.printf("%nBatch size: ");
			try {
				String str = br.readLine();
				int response = Integer.parseInt(str);
				if (response > 0) {
					batchSize = response;
					break;
				}
			}
			catch (IOException e) {
				e.printStackTrace();
			}
			catch (NumberFormatException e) {
				
			}
			System.out.printf("%nInvalid input. Please input a positive integer.");
		}
		
		// let the user select the number of epochs
		while (true) {
			// ensure the number of epochs is a positive integer, or tell the user the input is invalid and repeat
			System.out.printf("%nNumber of epochs: ");
			try {
				String str = br.readLine();
				int response = Integer.parseInt(str);
				if (response > 0) {
					epochs = response;
					break;
				}
			}
			catch (IOException e) {
				e.printStackTrace();
			}
			catch (NumberFormatException e) {
				
			}
			System.out.printf("%nInvalid input. Please input a positive integer.");
		}
		
		// let the user select the learning rate
		while (true) {
			// ensure the learning rate is a positive float, or tell the user the input is invalid and repeat
			System.out.printf("%nLearning rate: ");
			try {
				String str = br.readLine();
				float response = Float.parseFloat(str);
				if (response > 0) {
					learningRate = response;
					break;
				}
			}
			catch (IOException e) {
				e.printStackTrace();
			}
			catch (NumberFormatException e) {
				
			}
			System.out.printf("%nInvalid input. Please input a positive float.");
		}
		
		clearScreen();
		
		// train the network		
		nn.train(batchSize, epochs, -1, learningRate, trainingSet);

		networkTrained = true;
		
		enterToReturn();
		return true;
	}
	
	static final Command viewNetworkLayoutCmd = new Command(TerminalInterface::viewNetworkLayout, "View current network layout");
	static boolean viewNetworkLayout() {
		// let the user view the network layout
		clearScreen();
		// print the neural network
		System.out.printf(""+nn+"%n%n");
		enterToReturn();
		return true;
	}
	
	static final Command extrasMenuCmd = new Command(TerminalInterface::extrasMenu, "Extras");
	static boolean extrasMenu() {
		// display extras menu
		commandMenu(extraCommands, "Extras Menu:", "Back");
		return true;
	}
	
	static boolean continueOrQuit() {
		// prompt the user to decide to continue or quit
		System.out.printf("Press enter to continue, or type 0 and press enter to quit: ");
		try {
			if (br.readLine().startsWith("0")) {
				return false;
			}
			else {
				return true;
			}
		}
		catch (IOException e) {
			if (DEBUG) e.printStackTrace();
			return false;
		}
	}
	
	static boolean enterToReturn() {
		// prompt the user to hit enter to return
		System.out.printf("Press enter to return.");
		try {
			br.readLine();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	static boolean clearScreen() {
		// put a break in the screen
		// change string to change break size
		System.out.printf("%n%n");
		return true;
	}
	
	// main menu commands if untrained
	static final Command[] untrainedCommands = {
			defaultTrainCmd,
			loadCmd,
			extrasMenuCmd
	};
	
	// main menu commands if trained
	static final Command[] trainedCommands = {
			defaultTrainCmd, 
			loadCmd,
			displayTrainingAccuracyCmd,
			displayTestingAccuracyCmd,
			saveNetworkCmd,
			displayImagesCmd,
			displayIncorrectImagesCmd,
			extrasMenuCmd
	};
	
	// extra menu commands
	static final Command[] extraCommands = {
			viewNetworkLayoutCmd,
			customTrainCmd,
			customNetworkCmd
	};
}
