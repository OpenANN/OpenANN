from openann import *
import numpy

if __name__ == "__main__":
    # Create dataset
    X = numpy.array([[0, 1], [0, 0], [1, 1], [1, 0]])
    Y = numpy.array([[1], [0], [0], [1]])
    D = X.shape[1]
    F = Y.shape[1]
    N = X.shape[0]
    dataset = Dataset(X, Y)

    # Make the result repeatable
    RandomNumberGenerator().seed(0)

    # Create network
    net = Net()
    net.input_layer(D)
    net.fully_connected_layer(3, Activation.LOGISTIC)
    net.output_layer(F, Activation.LOGISTIC)

    # Train network
    stop_dict = {"minimal_value_differences" : 1e-10}
    lma = LMA(stop_dict)
    lma.optimize(net, dataset)

    # Use network
    for n in range(N):
        y = net.predict(X[n])
        print(y)
