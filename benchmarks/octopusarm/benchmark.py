import glob
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib
import zipfile

DIRECTORY = "octopus-code-distribution"
ARCHIVE = "octopus-code-distribution.zip"
URL = "http://www.cs.mcgill.ca/~dprecup/workshops/ICML06/Octopus/%s" % ARCHIVE

setup = {
  "env_dir" : DIRECTORY + "/environment/",
  "settings_dir" : "./",
  "cmd_mod" : " > /dev/null 2> /dev/null < /dev/null",
  "episodes" : 3,#1000,
  "gui" : False,
  "hidden_units" : 10
}
param_configs = [0, 5, 10, 20, 40, 80, 107]
runs = 1#10


def print_usage():
    print("Usage:")
    print("  python benchmark [download] [run] [evaluate]")


def download_octopusarm():
    if os.path.exists(DIRECTORY):
        print("Download is not required.")
        return

    print("Downloading %s" % URL)
    downloader = urllib.urlopen(URL)
    open(ARCHIVE, "w").write(downloader.read())

    print("Decompressing...")
    zipfile.ZipFile(ARCHIVE).extractall()
    os.remove(ARCHIVE)


def run_octopusarm():
    threads = [threading.Thread(target=start_run,
               kwargs={"parameters" : p}) for p in param_configs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(2)
    # TODO only works on linux!
    subprocess.call(["pkill", "java"])
    subprocess.call(["mkdir", "logs"])
    for f in glob.iglob("Neuro*.log"):
        shutil.move(f, "logs/"+f)
    print("Done.")


def start_run(parameters):
    port = 10000+parameters
    # The server will be used for all runs with this configuration
    server_thread = threading.Thread(target=start_server,
                                     kwargs={"port" : port})
    server_thread.start()
    for run in range(runs):
        time.sleep(2)
        print("Starting run %d with %d parameters." % (run+1, parameters))
        start_client(port, parameters)
        print("Finished run %d with %d parameters." % (run+1, parameters))
    server_thread.join(timeout=1)


def start_server(port, verbose=False):
    cmd = ["java",
           # The environment is written for Java 1.5, this will guarantee
           # compatibility with Java 1.6 and above:
           "-Djava.endorsed.dirs=" + setup["env_dir"] + "lib",
           "-jar", setup["env_dir"] + "octopus-environment.jar",
           # (De)activate GUI
           "external_gui" if setup["gui"] else "external",
           # Environment configuration
           setup["settings_dir"] + "settings.xml",
           str(port),
           # Suppress output
           setup["cmd_mod"]]
    if verbose:
        print(cmd)
    subprocess.call(cmd)


def start_client(port, parameters, verbose=False):
    cmd = ["./OctopusArmBenchmark", "localhost", str(port),
           str(setup["episodes"]), str(parameters),
           str(setup["hidden_units"]), setup["cmd_mod"]]
    if verbose:
        print(cmd)
    subprocess.call(cmd)


def evaluate_octopusarm(plot_axes):
    try:
        import pylab
        import numpy
    except ImportError:
        print("Required libraries: NumPy, Matplotlib.")
        exit(1)
    raise NotImplementedError()


if __name__ == "__main__":
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
