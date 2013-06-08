import os
import sys
import urllib
import numpy


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
CHAR_MATRIX = numpy.array([[char for char in row] for row in CHAR_MATRIX])


def print_usage():
    print("Usage:")
    print("  python benchmark directory [download] [run]")


def download_p300speller(directory):
    if not os.path.exists(directory + os.sep + "Subject_A_Train_Flashing.txt"):
        print("Please register at http://www.bbci.de/competition/iii\n"
            "to download the data set II from the BCI competition III.")

    if not os.path.exists(directory + os.sep + LABEL_FILES[0]):
        for i in range(len(LABEL_URLS)):
          print("Downloading %s" % LABEL_URLS[i])
          downloader = urllib.urlopen(LABEL_URLS[i])
          open(directory + os.sep + LABEL_FILES[i], "w").write(downloader.read())

    if not os.path.exists(directory + os.sep + "Subject_A_Test_StimulusType.txt"):
        convert_test_targets()

def convert_test_targets():
    print("Converting test targets...")
    for subject in ["a", "b"]:
        f = directory + os.sep + label_file_template % subject

        # Write Subject_A/B_Test_TargetChar.txt
        target_chars = open(f, "r").read().strip()
        target_ascii = [str(ord(c)) for c in target_chars]
        out = directory + os.sep + "Subject_%s_Test_TargetChar.txt" % subject.capitalize()
        open(out, "w").write("\n".join(target_ascii))

        # Write Subject_A/B_Test_StimulusType.txt
        f = open(directory + os.sep + "Subject_%s_Test_StimulusCode.txt" % subject.capitalize(), "r")
        stimulus_code = numpy.array([map(int, l.strip().split())
                                     for l in f.readlines()])
        out = directory + os.sep + "Subject_%s_Test_StimulusType.txt" % subject.capitalize()
        stimulus_type = open(out, "w")
        for t in range(stimulus_code.shape[0]):
            for epoch in range(stimulus_code.shape[1]):
                if stimulus_code[t, epoch] == 0:   # No stimulus
                    stimulus_type.write("0\t")
                elif stimulus_code[t, epoch] <= 6: # Column activated
                    col = stimulus_code[t, epoch] - 1
                    if target_chars[epoch] in CHAR_MATRIX[:, col]:
                        stimulus_type.write("1\t")
                    else:
                        stimulus_type.write("0\t")
                else:                              # Row activated
                    row = stimulus_code[t, epoch] - 7
                    if target_chars[epoch] in CHAR_MATRIX[row]:
                        stimulus_type.write("1\t")
                    else:
                        stimulus_type.write("0\t")
            stimulus_type.write("\n")
        stimulus_type.close()
    print("Done.")


def run_p300speller(directory):
    pass


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print_usage()
        exit(1)

    directory = sys.argv[1]

    for command in sys.argv[2:]:
        if command == "download":
            download_p300speller(directory)
        elif command == "run":
            run_p300speller(directory)
        else:
            print_usage()
            exit(1)
