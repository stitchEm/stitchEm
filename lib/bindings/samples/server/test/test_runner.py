import os
import subprocess
from optparse import OptionParser

TEST_MAIN = "test_main.py"


def parse_args():
    usage = "usage : %prog [options]"
    parser = OptionParser(usage)

    parser.add_option("-d",
                      "--lib-path",
                      type="string",
                      dest="lib_path",
                      help="points to library directory")

    (options, args) = parser.parse_args()

    return options, args


def run_tests():
    options, args = parse_args()
    if options.lib_path is None:
        print("You haven't specified path to the libraries and modules, please do so to proceed")
        return

    subprocess_env = os.environ.copy()
    subprocess_env["LD_LIBRARY_PATH"] = options.lib_path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    subprocess_env["PYTHONPATH"] = options.lib_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    subprocess.check_call(["/usr/bin/python", TEST_MAIN] + args, stderr=subprocess.STDOUT, env=subprocess_env)


if __name__ == "__main__":
    run_tests()
