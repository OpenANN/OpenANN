import sys
import subprocess

def print_usage():
    print("Usage:")
    print("  python benchmark [run]")


def run_pole_balancing():
    print("Starting benchmark, this will take some minutes...")
    subprocess.call("./PoleBalancingBenchmark")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()

    for command in sys.argv[1:]:
        if command == "run":
            run_pole_balancing()
        else:
            print_usage()
            exit(1)
