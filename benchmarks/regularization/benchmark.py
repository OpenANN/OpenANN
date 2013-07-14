import sys
import numpy
import pylab
from openann import *


def print_usage():
    print("Usage:")
    print("  python benchmark [run]")


def generate(N, D, offset):
    return numpy.random.randn(N, D) * 0.75 + numpy.ones(D) * offset

def generate_data(N, D):
    X1 = generate(N/2, D, numpy.zeros(D))
    X2 = generate(N/2, D, numpy.ones(D))
    C1 = [0]*(N/2)
    C2 = [1]*(N/2)
    return numpy.vstack((X1, X2)), numpy.hstack((C1, C2))

def enhance_data(X, T, multiplier):
    N = X.shape[0]
    X_t = numpy.ndarray((N*multiplier, X.shape[1]))
    T_t = numpy.ndarray((N*multiplier, T.shape[1]))
    for i in range(multiplier):
        for n in range(N):
            X_t[i*N+n] = X[n] + numpy.random.randn(2)*0.2
            T_t[i*N+n] = T[n]
    X = X_t
    T = T_t
    return X, T

def build_net(D, H, F, l2_penalty=None, C=None, comp=""):
    net = Net()
    if l2_penalty is not None:
        net.set_regularization(l2_penalty=l2_penalty)
    net.input_layer(D)
    if C is None:
        for h in H:
            net.fully_connected_layer(h, Activation.TANH)
        net.output_layer(F, Activation.LINEAR)
    else:
        net.fully_connected_layer(H[0], Activation.TANH)
        net.compressed_layer(H[1], C[0], Activation.TANH, comp)
        net.compressed_output_layer(F, C[1], Activation.LINEAR, comp)
    return net


def run_regularization():
    numpy.random.seed(0)

    regularization = ["none", "simple", "compression", "l2", "noise"][2]
    print(regularization)

    N = 200
    Nt = 1000
    D = 2
    F = 2
    n_folds = 5
    X, C = generate_data(N, D)
    Xt, Ct = generate_data(Nt, D)

    # 1-of-c encoding
    T = numpy.zeros((N, F))
    T[range(N), C] = 1.0
    Tt = numpy.zeros((Nt, F))
    Tt[range(Nt), Ct] = 1.0

    if regularization == "none":
        net = build_net(D, [50, 50], F)
    elif regularization == "simple":
        net = build_net(D, [5, 5], F)
    elif regularization == "compression":
        net = build_net(D, [50, 50], F, C=[2, 2], comp="gaussian")
    elif regularization == "l2":
        net = build_net(D, [50, 50], F, l2_penalty=0.1)
    elif regularization == "noise":
        X, T = enhance_data(X, T, 5)
        net = build_net(D, [50, 50], F)

    net.set_error_function(Error.CE)
    opt = CG({"maximal_iterations" : 1000})
    ts = Dataset(X, T)
    tes = Dataset(Xt, Tt)
    Log.set_info()
    print("%.2f %%" % (100*cross_validation(n_folds, net, ts, opt)))
    net.initialize()
    opt.optimize(net, ts)
    print("%.2f %%" % (100*accuracy(net, tes)))

    XX, YY = numpy.meshgrid(numpy.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                            numpy.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    Z = numpy.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            Z[i, j] = net.predict([XX[i, j], YY[i, j]])[0, 0]

    pylab.contourf(XX, YY, Z)
    pylab.plot(X[numpy.nonzero(C == 0), 0],
               X[numpy.nonzero(C == 0), 1], "bo")
    pylab.plot(X[numpy.nonzero(C == 1), 0],
               X[numpy.nonzero(C == 1), 1], "ro")
    pylab.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "run":
            run_regularization()
        else:
            print_usage()
            exit(1)
