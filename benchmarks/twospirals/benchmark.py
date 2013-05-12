import sys
import subprocess

def print_usage():
    print("Usage:")
    print("  python benchmark [run]")


def run_two_spirals():
    print("Starting benchmark, this will take about one hour...")
    subprocess.call("./TwoSpiralsBenchmark")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "run":
            run_two_spirals()
        else:
            print_usage()
            exit(1)
