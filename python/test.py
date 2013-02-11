from openann import *
import numpy
import pylab

d = 1
h = 100
f = 1
n = 100

inputs = numpy.ndarray((d, n))
outputs = numpy.ndarray((d, n))
for i in range(n):
  x = i*2*numpy.pi/n
  inputs[:, i] = x
  outputs[:, i] = numpy.cos(x)

net = DeepNetwork()
net.input_layer(d, True, 0.05)
#net.alpha_beta_filter_layer(0.1)
#net.convolutional_layer(5, 3, 3, "tanh")
#net.subsampling_layer(2, 2, "rectifier")
#net.maxpooling_layer(2, 2)
#net.compressed_layer(10, 5, "tanh", "sparse")
net.fully_connected_layer(h, "rectifier", True)
net.output_layer(f, "linear")
net.training_set(inputs, outputs)
net.train("lma", "sse", {"maximalIterations" : 10, "minimalValueDifferences" : 1e-8})

pylab.plot(inputs[0, :], net.predict(inputs)[0, :])
pylab.plot(inputs[0, :], outputs[0, :])
pylab.show()

