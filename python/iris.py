try:
    from sklearn import datasets
except:
    print("scikit-learn is required to run this example.")
    exit(1)
from openann import *

# Load IRIS dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target
D = X.shape[1]
F = len(numpy.unique(Y))
N = len(X)

# Preprocess data (normalization and 1-of-c encoding)
X = (X - X.mean(axis=0)) / X.std(axis=0)
T = numpy.zeros((N, F))
T[(range(N), Y)] = 1.0

net = Net()
net.set_regularization(0.0, 0.01, 0.0)
net.input_layer(D)
net.fully_connected_layer(100, Activation.RECTIFIER)
net.fully_connected_layer(100, Activation.RECTIFIER)
net.output_layer(F, Activation.LINEAR)
net.set_error_function(Error.CE)

# Split dataset into training set and validation set and make sure that
# each class is equally distributed in the datasets
X1 = numpy.vstack((X[0:40], X[50:90], X[100:140]))
T1 = numpy.vstack((T[0:40], T[50:90], T[100:140]))
training_set = Dataset(X1, T1)
X2 = numpy.vstack((X[40:50], X[90:100], X[140:150]))
T2 = numpy.vstack((T[40:50], T[90:100], T[140:150]))
validation_set = Dataset(X2, T2)

# Train for 500 episodes (with tuned parameters for MBSGD)
optimizer = MBSGD({"maximal_iterations": 500}, learning_rate=0.7,
    learning_rate_decay=0.999, min_learning_rate=0.001, momentum=0.5,
    batch_size=16)
optimizer.optimize(net, training_set)

print("")
print("Iris data set has %d inputs, %d classes and %d examples" % (D, F, N))
print("The data has been split up input training and validation set.")
print("Correct predictions on training set: %d/%d"
      % (classification_hits(net, training_set), len(X1)))
print("Correct predictions on test set: %d/%d"
      % (classification_hits(net, validation_set), len(X2)))
