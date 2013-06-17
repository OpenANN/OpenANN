from openann import *
import numpy
import pylab

# Create network
net = Net()
net.set_regularization(0.0, 0.0001, 0.0)
net.input_layer(1)
net.fully_connected_layer(10, Activation.LOGISTIC)
net.fully_connected_layer(10, Activation.LOGISTIC)
net.output_layer(1, Activation.LINEAR)

# Create dataset
X = numpy.linspace(0, 2*numpy.pi, 500)[:, numpy.newaxis]
T = numpy.sin(X) + numpy.random.randn(*X.shape) * 0.1
dataset = Dataset(X, T)

Log.info("Using %d samples with %d inputs and %d outputs"
    % (dataset.samples(), dataset.inputs(), dataset.outputs()))

# Train network
stop  = {
    "maximal_iterations": 50,
    "minimal_value_differences": 1e-8
}
lma = LMA(stop)
lma.optimize(net, dataset)

# Predict data
Y = net.predict(X)

# Plot dataset and hypothesis
pylab.plot(X, T, ".", label="Data Set")
pylab.plot(X, Y, label="Prediction", linewidth=3)
pylab.legend()
pylab.show()


