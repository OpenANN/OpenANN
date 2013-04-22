"""Simple example for OpenANN Python bindings.
"""
from openann import *
import numpy
import pylab


if __name__ == "__main__":
  # Input dimension
  d = 1
  # Hidden nodes
  h = 200
  # Output dimension
  f = 1

  # Size of training set
  n = 500
  # Standard deviation of noise in the training set
  noise_std_dev = 0.1

  # Create noisy sine
  # Each column of these matrices contains an instance
  inputs = numpy.atleast_2d(numpy.linspace(0, 2*numpy.pi, n))
  outputs = numpy.sin(numpy.random.normal(inputs, numpy.ones(n)*noise_std_dev))

  # Setup network
  net = Net()
  # Your input data can have up to 3 dimensions, e. g. an image could have
  # color channels x rows x columns: (3, 16, 16). Convolutional layers or
  # pooling layers require 3 dimensions!
  net.input_layer(d, True, 0.05)
  # Choose one activation function from
  # logistic, tanh, tanhscaled, rectifier, linear
  net.fully_connected_layer(h, "rectifier", True)
  # The last layer has to be either an output_layer or a compressed_output_layer
  net.output_layer(f, "linear")

  # You can use any of these layers in your network:
  #net.alpha_beta_filter_layer(0.1)
  #net.convolutional_layer(5, 3, 3, "tanh")
  #net.subsampling_layer(2, 2, "rectifier")
  #net.maxpooling_layer(2, 2)
  #net.local_response_normalization_layer(1, 5, 1e-4, 0.75, True)
  #net.fully_connected_layer(10, "rectifier", True)
  #net.compressed_layer(10, 5, "tanh", "sparse")
  #net.extreme_layer(100, "rectifier", True)
  #net.dropout_layer(0.5)

  net.training_set(inputs, outputs)
  # Define stopping criteria, e. g.
  # maximalFunctionEvaluations, maximalIterations, maximalRestarts,
  # minimalValue, minimalValueDifferences, minimalSearchSpaceStep
  # Note that not all optimzers use all stopping criteria!
  stop_dict = {"maximalIterations" : 10,
               "minimalValueDifferences" : 1e-8}
  # Train the parameters of the network, i. e. minimize Sum of Squared Errors,
  # available error functions are:
  # sse, mse, ce
  # Available optimization algorithms are
  # LMA, MBSGD, CMAES
  net.train("LMA", "sse", stop_dict)

  # Plot actual data and prediction
  pylab.title("SSE = %.2f" % net.error())
  pylab.plot(inputs[0], outputs[0], ".", label="Data Set")
  pylab.plot(inputs[0], net.predict(inputs)[0], label="Prediction")
  pylab.legend()
  pylab.show()
