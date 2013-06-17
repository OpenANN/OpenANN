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
    learning_rate=0.7, learning_rate_decay=0.99897, min_learning_rate=0.00001,
    momentum=0.5,
    batch_size=16)

net.set_regularization(0.001, 0.0, 0.0)
net.input_layer(D)
net.fully_connected_layer(200, Activation.RECTIFIER)
net.fully_connected_layer(200, Activation.RECTIFIER)
net.output_layer(F, Activation.LINEAR)
net.set_error_function(Error.CE)

X1 = numpy.vstack((X[0:40], X[50:90], X[100:140]))
T1 = numpy.vstack((T[0:40], T[50:90], T[100:140]))
X2 = numpy.vstack((X[40:50], X[90:100], X[140:150]))
T2 = numpy.vstack((T[40:50], T[90:100], T[140:150]))
training_set = Dataset(X1, T1)
test_set = Dataset(X2, T2)

optimizer.optimize(net, training_set)

print classification_hits(net, training_set)
print classification_hits(net, test_set)
