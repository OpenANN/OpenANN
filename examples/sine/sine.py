## \page Sine Sine
#
# \section DataSet Data Set
#
# In this example, a sine function will be approximated from noisy measurements.
# This is an example for nonlinear regression. To run this example, you have
# to install matplotlib. It is a plotting library for Python.
#
# \section Code
#
# \include "sine/sine.py"

try:
    import pylab
except:
    print("Matplotlib is required")
    exit(1)
from openann import *
import numpy

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
