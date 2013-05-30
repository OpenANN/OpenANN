import glob
import os
import shutil
import sys
import subprocess
import tarfile
import urllib

URL = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
ARCHIVE_NAME = "cifar-10-binary.tar.gz"
DATA_FOLDER = "cifar-10-batches-bin"
DATA_FILES = ["data_batch_%d.bin" % i for i in range(1, 6)]
DATA_FILES.append("test_batch.bin")


def print_usage():
    print("Usage:")
    print("  python benchmark [download] [run] [evaluate]")


def download_cifar10():
    if all(os.path.exists(f) for f in DATA_FILES):
        print("Download is not required.")
        return

    print("Starting download...")
    downloader = urllib.urlopen(URL)
    open(ARCHIVE_NAME, "w").write(downloader.read())

    print("Decompressing...")
    tarfile.open(ARCHIVE_NAME, "r:gz").extractall()

    print("Move files...")
    for f in DATA_FILES:
        shutil.copy(DATA_FOLDER + "/" + f, f)

    print("Cleaning up...")
    os.remove(ARCHIVE_NAME)
    shutil.rmtree(DATA_FOLDER)

    print("Done.")


def run_cifar10():
    subprocess.call("./CIFAR10")


def evaluate_cifar10(plot_axes):
    try:
        import pylab
        import numpy
    except ImportError:
        print("Required libraries: NumPy, Matplotlib.")
        exit(1)

    axes = ["Epoch", "MSE", "Correct", "Errors", "Time"]

    log = []
    for f in glob.iglob("evaluation-*.log"):
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
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "download":
            download_cifar10()
        elif command == "run":
            run_cifar10()
        elif command == "evaluate":
            evaluate_cifar10(plot_axes=["Time", "Errors"])
        else:
            print_usage()
            exit(1)
