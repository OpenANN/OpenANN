## \page IrisBenchmark Iris Flower Dataset
#
#  The iris dataset is a standard machine learning dataset.
#  See e.g. the <a href="http://en.wikipedia.org/wiki/Iris_flower_data_set"
#  target=_blank>Wikipedia article</a> for more details.
#
#  You can start the benchmark with the script:
#  \verbatim
#  python benchmark.py [run]
#  \endverbatim
#  Note that you need Scikit Learn to load the dataset.
#
#  The result will look like
#  \verbatim
#  Iris data set has 4 inputs, 3 classes and 150 examples
#  The data has been split up input training and validation set.
#  Correct predictions on training set: 120/120
#  Confusion matrix:
#  [[ 40.   0.   0.]
#  [  0.  40.   0.]
#  [  0.   0.  40.]]
#  Correct predictions on test set: 30/30
#  Confusion matrix:
#  [[ 10.   0.   0.]
#  [  0.  10.   0.]
#  [  0.   0.  10.]]
#  \endverbatim

import sys
try:
    from sklearn import datasets
except:
    print("scikit-learn is required to run this example.")
    exit(1)
try:
    from openann import *
except:
    print("OpenANN Python bindings are not installed!")
    exit(1)


def print_usage():
    print("Usage:")
    print("  python benchmark [run]")


def run_iris():
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

    # Setup network
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
    Log.set_info() # Deactivate debug output
    optimizer.optimize(net, training_set)

    print("Iris data set has %d inputs, %d classes and %d examples" % (D, F, N))
    print("The data has been split up input training and validation set.")
    print("Correct predictions on training set: %d/%d"
          % (classification_hits(net, training_set), len(X1)))
    print("Confusion matrix:")
    print(confusion_matrix(net, training_set))
    print("Correct predictions on test set: %d/%d"
          % (classification_hits(net, validation_set), len(X2)))
    print("Confusion matrix:")
    print(confusion_matrix(net, validation_set))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "run":
            run_iris()
        else:
            print_usage()
            exit(1)
