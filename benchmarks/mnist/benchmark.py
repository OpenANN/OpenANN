FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
ARCHIVES = ["%s.gz" % f for f in FILES]
URLS = ["http://yann.lecun.com/exdb/mnist/%s" % a for a in ARCHIVES]


def print_usage():
    print("Usage:")
    print("  python benchmark [download] [run] [evaluate]")


def download_mnist():
    import urllib
    import gzip
    import os

    if all(os.path.exists(f) for f in FILES):
        print("Download is not required.")
        return

    for i in range(len(URLS)):
        print("Downloading %s" % URLS[i])
        downloader = urllib.urlopen(URLS[i])

        with open(ARCHIVES[i], "wb") as out:
            while True:
                data = downloader.read(1024)
                if len(data) == 0: break
                out.write(data)

        archive = gzip.open(ARCHIVES[i])
        open(FILES[i], "w").write(archive.read())
        os.remove(ARCHIVES[i])


def run_mnist():
    import subprocess
    subprocess.call("./MNIST")


def evaluate_mnist(plot_axes):
    try:
        import pylab
        import numpy
    except ImportError:
        print("Required libraries: NumPy, Matplotlib.")
        exit(1)
    import os
    import glob

    axes = ["Epoch", "MSE", "Correct", "Errors", "Time"]

    log = []
    for f in glob.iglob("dataset-*.log"):
        run = []
        for l in open(f, "r").readlines():
            l = l.strip()
            if len(l) > 0 and l[0] != "#":
                run.append(map(float, l.split()))
        log.append(numpy.array(run).T)

    for run in log:
        if run.size > 0:
            pylab.plot(run[axes.index(plot_axes[0])],
                       run[axes.index(plot_axes[1])])
    pylab.xlabel(plot_axes[0])
    pylab.ylabel(plot_axes[1])
    pylab.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "download":
            download_mnist()
        elif command == "run":
            run_mnist()
        elif command == "evaluate":
            evaluate_mnist(plot_axes=["Time", "Errors"])
        else:
            print_usage()
            exit(1)
