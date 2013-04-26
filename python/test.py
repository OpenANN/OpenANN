from openann import *
from numpy import *
import pylab

stop  = {
    "maximal_iterations": 50,
    "minimal_value_differences": 1e-8
}

net   = Net()
lma   = LMA(stop)

net.input_layer(1, 1)
net.fully_connected_layer(200, Activation.LOGISTIC)
net.output_layer(1, Activation.LINEAR)

inputs = numpy.atleast_2d(numpy.linspace(0, 2*numpy.pi, 500))
outputs = numpy.sin(numpy.random.normal(inputs, 0.1))

dataset = Dataset(inputs, outputs)

Log.info("Using {0} samples with {1} inputs and {2} outputs".
    format(dataset.samples(), dataset.inputs(), dataset.outputs()))

lma.optimize(net, dataset)

prediction = [net.predict(inputs[:, i])[0]
              for i in range(inputs.shape[1])]

pylab.plot(inputs[0], outputs[0], ".", label="Data Set")
pylab.plot(inputs[0], prediction, label="Prediction")
pylab.legend()
pylab.show()


