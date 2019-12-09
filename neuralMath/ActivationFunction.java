package neuralMath;
import java.util.function.Function;

public class ActivationFunction {
	// variables of activation functions
	Function<Double, Double> activationFunction;
	Function<Double, Double> derivativeFunction;
	String name;

	// common activation functions
	public static final ActivationFunction sigmoid = new ActivationFunction(z -> 1.0/(1.0+Math.exp(-z)), z -> z * (1.0 - z), "sigmoid");
	public static final ActivationFunction ReLU = new ActivationFunction(z -> (z > 0) ? z : 0.0, z -> (z > 0) ? 1.0 : 0.0, "ReLU");
	public static final ActivationFunction leakyReLU = new ActivationFunction(z -> (z > 0) ? z : 0.01 * z, z -> (z > 0) ? 1.1 : 0.01, "leakyReLU");
	public static final ActivationFunction tanh = new ActivationFunction(z -> (2.0/(1.0+Math.exp(-2.0*z)))-1.0, z -> 1.0 - z*z, "tanh");
	
	// list of these activation functions
	public static final ActivationFunction[] list = {sigmoid, tanh, ReLU, leakyReLU};
	
	// constructor
	public ActivationFunction(Function<Double, Double> activationFunction, Function<Double, Double> derivativeFunction, String name) {
		this.activationFunction = activationFunction;
		this.derivativeFunction = derivativeFunction;
		this.name = name;
	}
	
	// method to get an activation function by its name as a string
	public static ActivationFunction fromString(String activationFuncType) {
		switch(activationFuncType)
		{
		case "sigmoid":
			return sigmoid;
		case "tanh":
			return tanh;
		case "ReLU":
			return ReLU;
		case "leakyReLU":
			return leakyReLU;
		}
		return null;
	}
	
	// activation function
	public double activate(double z) {
		return this.activationFunction.apply(z);
	}
	
	// derivative of activation function
	public double derivative(double z) {
		return this.derivativeFunction.apply(z);
	}
	
	// string name
	public String toString() {
		return name;
	}
}
