## \page SARCOSBenchmark SARCOS Inverse Dynamics Problem
#
#  The SARCOS dataset is taken from
#  <a href="http://www.gaussianprocess.org/gpml/data/" target=_blank>this</a>
#  website. This is an inverse dynamics problem, i.e. we have to predict the
#  7 joint torques given the joint positions, velocities and accelerations.
#  Hence, we have to solve a regression problem with 21 inputs and 7 outputs
#  and a very nonlinear function.
#
#  The optimization problem is very hard. Underfitting is a much bigger
#  problem than overfitting. For this reason, we need a very big network
#  that has four hidden layers with 200 nodes each. The deep architecture
#  makes the optimization problem very hard but it is more efficient than a
#  shallow network. However, we can do two things to increase the optimization
#  speed drastically: we use a non-saturating activation function (rectified
#  linear units) and mini-batch stochastic gradient descent.
#
#  You can start the benchmark with the script:
#  \verbatim
#  python benchmark.py [download] [run]
#  \endverbatim
#  Note that you need SciPy to load the dataset and matplotlib to display some
#  results.
#
#  The output will look like
#  \verbatim
#  Dimension 1: nMSE = 0.938668% (training) / 0.903342% (validation)
#  Dimension 2: nMSE = 0.679012% (training) / 0.647091% (validation)
#  Dimension 3: nMSE = 0.453497% (training) / 0.442720% (validation)
#  Dimension 4: nMSE = 0.242476% (training) / 0.240360% (validation)
#  Dimension 5: nMSE = 1.010049% (training) / 1.044068% (validation)
#  Dimension 6: nMSE = 0.851110% (training) / 0.796895% (validation)
#  Dimension 7: nMSE = 0.474232% (training) / 0.465929% (validation)
#  \endverbatim
#  You see the normalized mean squared error (nMSE) for each output dimension
#  on the training set and the test set. The nMSE is the mean squared error
#  divided by the variance of the corresponding output dimension. In addition,
#  a plot that compares the actual and the predicted output of one dimension
#  will occur.

import os
import sys
import urllib
try:
    import scipy.io
except:
    print("SciPy is required for this benchmark.")
    exit(1)
try:
    from openann import *
except:
    print("OpenANN Python bindings are not installed!")
    exit(1)


FILES = ["sarcos_inv.mat", "sarcos_inv_test.mat"]
URLS = ["http://www.gaussianprocess.org/gpml/data/%s" % f for f in FILES]


def print_usage():
    print("Usage:")
    print("  python benchmark [download] [run]")


def download_sarcos():
    if all(os.path.exists(f) for f in FILES):
        print("Download is not required.")
        return

    for i in range(len(URLS)):
        print("Downloading %s" % URLS[i])
        downloader = urllib.urlopen(URLS[i])

        with open(FILES[i], "wb") as out:
            while True:
                data = downloader.read(1024)
                if len(data) == 0: break
                out.write(data)


def run_sarcos():
    print("Loading dataset...")
    a = scipy.io.loadmat("sarcos_inv.mat")
    X = a["sarcos_inv"][:, :21]
    Y = a["sarcos_inv"][:, 21:]
    b = scipy.io.loadmat("sarcos_inv_test.mat")
    Xtest = b["sarcos_inv_test"][:, :21]
    Ytest = b["sarcos_inv_test"][:, 21:]
    print("Starting benchmark, this will take some minutes...")

    # Normalize data
    n = Normalization()
    X = n.fit(X).transform(X)
    Xtest = n.transform(Xtest)
    Y = n.fit(Y).transform(Y)
    Ytest = n.transform(Ytest)

    training_set = DataSet(X, Y)
    validation_set = DataSet(Xtest, Ytest)

    D = X.shape[1]
    F = Y.shape[1]
    net = Net()
    net.input_layer(D)
    net.fully_connected_layer(400, Activation.RECTIFIER)
    net.fully_connected_layer(200, Activation.RECTIFIER)
    net.fully_connected_layer(200, Activation.RECTIFIER)
    net.output_layer(F, Activation.LINEAR)

    stop_dict = {"maximal_iterations" : 100}
    opt = MBSGD(stop_dict, learning_rate=0.2, learning_rate_decay=0.9999,
                min_learning_rate=0.001, momentum=0.5, batch_size=128,
                nesterov=False)
    opt.optimize(net, training_set)

    pred = net.predict(X)
    pred_test = net.predict(Xtest)
    var = Y.var(axis=0) # in case we do not normalize the outputs
    for f in range(F):
        nMSE_train = sum((Y[:, f] - pred[:, f])**2) / len(Y) / var[f]
        nMSE_test = sum((Ytest[:, f] - pred_test[:, f])**2) / len(Ytest) / var[f]
        print("Dimension %d: nMSE = %f%% (training) / %f%% (validation)"
              % (f+1, nMSE_train*100, nMSE_test*100))

    try:
        import pylab
    except:
        print("Cannot plot the result. Matplotlib is not available.")
        exit(1)

    dim = 0
    n_samples = 200
    pylab.plot(Ytest[:n_samples, dim], label="Actual")
    pylab.plot(net.predict(Xtest[:n_samples])[:, dim], label="Predicted")
    pylab.legend(loc="best")
    pylab.title("Output of %d samples from dimension %d (validation set)"
                % (n_samples, dim+1))
    pylab.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "download":
            download_sarcos()
        elif command == "run":
            run_sarcos()
        else:
            print_usage()
            exit(1)
