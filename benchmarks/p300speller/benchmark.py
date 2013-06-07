import sys


def print_usage():
    print("Usage:")
    print("  python benchmark [download] [run] [evaluate]")


def download_p300speller():
    print("Please register at http://www.bbci.de/competition/iii\n"
        "to download the data set II from the BCI competition III.")


def run_p300speller():
    pass


def evaluate_p300speller():
    pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "download":
            download_p300speller()
        elif command == "run":
            run_p300speller()
        elif command == "evaluate":
            evaluate_p300speller()
        else:
            print_usage()
            exit(1)
