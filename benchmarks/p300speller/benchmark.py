import os
import sys
import urllib

LABEL_FILES = ["true_labels_%s.txt" % subject for subject in ["a", "b"]]
LABEL_URLS = ["http://www.bbci.de/competition/iii/results/albany/%s" % f
              for f in LABEL_FILES]

CHAR_MATRIX = [
    "ABCDEF",
    "GHIJKL",
    "MNOPQR",
    "STUVWX",
    "YZ1234",
    "56789_"
]

def print_usage():
    print("Usage:")
    print("  python benchmark directory [download] [run] [evaluate]")


def download_p300speller(directory):
    if not os.path.exists(directory + "/" + "Subject_A_Train_Flashing.txt"):
        print("Please register at http://www.bbci.de/competition/iii\n"
            "to download the data set II from the BCI competition III.")

    if not os.path.exists(directory + "/" + LABEL_FILES[0]):
        for i in range(len(LABEL_URLS)):
          print("Downloading %s" % LABEL_URLS[i])
          downloader = urllib.urlopen(LABEL_URLS[i])
          open(directory + "/" + LABEL_FILES[i], "w").write(downloader.read())

    if True:#not os.path.exists(directory + "/" + "Subject_A_Test_StimulusType.txt"):
        convert_test_targets()

def convert_test_targets():
    print("Converting test targets...")
    for f in [directory + "/" + f for f in LABEL_FILES]:
      print(f)


def run_p300speller(directory):
    pass


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print_usage()

    directory = sys.argv[1]

    for command in sys.argv[2:]:
        if command == "download":
            download_p300speller(directory)
        elif command == "run":
            run_p300speller(directory)
        else:
            print_usage()
            exit(1)
