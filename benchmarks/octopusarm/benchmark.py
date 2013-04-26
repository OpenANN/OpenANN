DIRECTORY = "octopus-code-distribution"
ARCHIVE = "octopus-code-distribution.zip"
URL = "http://www.cs.mcgill.ca/~dprecup/workshops/ICML06/Octopus/%s" % ARCHIVE


def print_usage():
    print("Usage:")
    print("  python benchmark [download] [run] [evaluate]")


def download_octopusarm():
    import urllib
    import os
    import zipfile

    if False:# os.path.exists(DIRECTORY):
        print("Download is not required.")
        return

    print("Downloading %s" % URL)
    downloader = urllib.urlopen(URL)
    open(ARCHIVE, "w").write(downloader.read())

    print("Decompressing...")
    zipfile.ZipFile(ARCHIVE).extractall()
    os.remove(ARCHIVE)


def run_octopusarm():
    import subprocess
    # TODO implement in Python
    subprocess.call("ruby run")


def evaluate_octopusarm(plot_axes):
    try:
        import pylab
        import numpy
    except ImportError:
        print("Required libraries: NumPy, Matplotlib.")
        exit(1)
    raise NotImplementedError()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "download":
            download_octopusarm()
        elif command == "run":
            run_octopusarm()
        elif command == "evaluate":
            evaluate_octopusarm(plot_axes=["Time", "Errors"])
        else:
            print_usage()
            exit(1)
