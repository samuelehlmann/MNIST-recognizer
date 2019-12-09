# MNIST-recognizer

I made this program for my AI class in Fall 2019. It implements neural networks, coded from scratch, without even an API for matrix operations. Its purpose is to recognize handwritten digits from the MNIST dataset, and to be used the training and testing sets should be downloaded from https://pjreddie.com/projects/mnist-in-csv/ and moved to the same directory as TerminalInterface.java, which is the entry point of the program. 

The stipulations for the assignment were that it should:
* be written
* implement a neural network that had one input layer for MNIST images, one hidden layer of 15 neurons, and one output layer of 10 neurons using a sigmoid activation function
* display the accuracy after training at each epoch
* be able to save and load trained networks from a single file
* be able to display accuracy on training data
* be able to display accuracy on testing data
* exit


Additionally, bonus points were made available if it could:
* run the network on the testing data and display each image, its classification, the output of the neural network, and if it was correct or not
* do the same as the previous point, but only display correct images


I did all of the previous, as well as extended this program of my own accord (receiving no bonus points for doing so) to:
* allow neural networks to be user defined, such that the user can
  * define the number of hidden layers
  * define the number of neurons for each hidden layer
  * define the activation function for each hidden layer separately, as well as
  * define the activation function for the final layer
* allow the user to train with custom parameters instead of preset parameters
* allow the user to save networks to separate files
* display a list of networks the user can load
* view the current network layout
