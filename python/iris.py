from sklearn import datasets
from openann import *

iris = datasets.load_iris()
X = iris.data
Y = iris.target
D = X.shape[1]
F = 3
N = len(X)

X = (X - X.mean(axis=0)) / X.std(axis=0)
T = numpy.zeros((N, F))
for n in range(N):
    T[n, Y[n]] = 1.0

net = Net()
stop  = {
    "maximal_iterations": 1000
}
optimizer = MBSGD(
    stop,
    learning_rate=0.5, learning_rate_decay=0.999, min_learning_rate=0.001,
    momentum=0.5,
    batch_size=16)

#net.set_regularization(0.0, 0.0001, 0.0)
net.input_layer(D)
net.fully_connected_layer(200, Activation.RECTIFIER)
net.fully_connected_layer(200, Activation.RECTIFIER)
net.output_layer(F, Activation.LINEAR)
net.set_error_function(Error.CE)

dataset = Dataset(X, T)

optimizer.optimize(net, dataset)

print classification_hits(net, dataset)
