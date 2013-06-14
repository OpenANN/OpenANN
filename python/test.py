from openann import *
import numpy
import pylab

stop  = {
    "maximal_iterations": 50,
    "minimal_value_differences": 1e-8
}

net = Net()
lma = LMA(stop)

net.input_layer(1)
net.set_regularization(0.0, 0.0001, 0.0)
net.fully_connected_layer(10, Activation.LOGISTIC)
net.fully_connected_layer(10, Activation.LOGISTIC)
net.output_layer(1, Activation.LINEAR)

inputs = numpy.linspace(0, 2*numpy.pi, 500)[:, numpy.newaxis]
outputs = numpy.sin(inputs) + numpy.random.randn(*inputs.shape) * 0.1

dataset = Dataset(inputs, outputs)

Log.info("Using {0} samples with {1} inputs and {2} outputs".
    format(dataset.samples(), dataset.inputs(), dataset.outputs()))

lma.optimize(net, dataset)

prediction = numpy.array([net.predict(inputs[i])
                          for i in range(inputs.shape[0])])

pylab.plot(inputs[:, 0], outputs[:, 0], ".", label="Data Set")
pylab.plot(inputs[:, 0], prediction[:, 0], label="Prediction")
pylab.legend()
pylab.show()


