import os
import sys
import urllib

label_file_template = "true_labels_%s.txt"
LABEL_FILES = [label_file_template % subject for subject in ["a", "b"]]
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
    if not os.path.exists(directory + os.sep + "Subject_A_Train_Flashing.txt"):
        print("Please register at http://www.bbci.de/competition/iii\n"
            "to download the data set II from the BCI competition III.")

    if not os.path.exists(directory + os.sep + LABEL_FILES[0]):
        for i in range(len(LABEL_URLS)):
          print("Downloading %s" % LABEL_URLS[i])
          downloader = urllib.urlopen(LABEL_URLS[i])
          open(directory + os.sep + LABEL_FILES[i], "w").write(downloader.read())

    if True:#not os.path.exists(directory + os.sep + "Subject_A_Test_StimulusType.txt"):
        convert_test_targets()

def convert_test_targets():
    print("Converting test targets...")
    for subject in ["a", "b"]:
      f = directory + os.sep + label_file_template % subject

      # Write Subject_A/B_Test_TargetChar.txt
      target_chars = open(f, "r").read().strip()
      target_ascii = [str(ord(c)) for c in target_chars]
      f = "dummy.txt" #directory + os.sep + "Subject_%s_Test_TargetChar.txt" % subject.capitalize()
      open(f, "w").write("\n".join(target_ascii))

      # Write Subject_A/B_Test_StimulusType.txt
      f = "dummy2.txt" #directory + os.sep + "Subject_%s_Test_StimulusType.txt" % subject.capitalize()
      stimulus_type = open(directory + os.sep + "Subject_%s_Test_StimulusCode.txt" % subject.capitalize(), "r")
      # TODO

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
