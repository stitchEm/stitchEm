import sys
import os
import unittest

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
SERVER_DIR = os.path.join(TEST_DIR, "..")
sys.path.insert(0, SERVER_DIR)

import utils.log

if __name__ == "__main__":
    utils.log.set_python_log(4)
    suite = unittest.TestLoader().discover(TEST_DIR)
    retcode = not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    sys.exit(retcode)
