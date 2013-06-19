import os
import sys
import urllib
try:
    import pylab
    import scipy.io
except:
    print("SciPy and matplotlib are required for this benchmark.")
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
    Xmu = X.mean(axis=0)
    Xstd = X.std(axis=0)
    X = (X - Xmu) / Xstd
    Xtest = (Xtest - Xmu) / Xstd
    Ymu = Y.mean(axis=0)
    Ystd = Y.std(axis=0)
    Y = (Y - Ymu) / Ystd
    Ytest = (Ytest - Ymu) / Ystd
    # Learn only one dimension
    #Y = Y[:, 0, numpy.newaxis]
    #Ytest = Ytest[:, 0, numpy.newaxis]

    training_set = Dataset(X, Y)
    validation_set = Dataset(Xtest, Ytest)

    D = X.shape[1]
    F = Y.shape[1]
    net = Net()
    net.input_layer(D)
    net.fully_connected_layer(250, Activation.TANH)
    net.fully_connected_layer(100, Activation.TANH)
    net.output_layer(F, Activation.LINEAR)

    stop_dict = {"maximal_iterations" : 100}
    opt = MBSGD(stop_dict, learning_rate=0.2, learning_rate_decay=0.9999,
                min_learning_rate=0.001, momentum=0.5, batch_size=128)
    opt.optimize(net, training_set)

    print("RMSE")
    print("Training:", rmse(net, training_set))
    print("Validation:", rmse(net, validation_set))

    pred = net.predict(X)
    pred_test = net.predict(Xtest)
    var = Y.var(axis=0) # in case we do not normalize the outputs
    for f in range(F):
        nMSE_train = sum((Y[:, f] - pred[:, f])**2) / len(Y) / var[f]
        nMSE_test = sum((Ytest[:, f] - pred_test[:, f])**2) / len(Ytest) / var[f]
        print("Dimension %d: nMSE = %f%% (training) / %f%% (validation)"
              % (f, nMSE_train*100, nMSE_test*100))

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
