import os

from algorithm.algorithm_runner import AlgorithmRunner
from utils.ptv import PTV

CURRENT_DIR = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
ALGORITHM_CONFIGURATION = os.path.join(CURRENT_DIR, "../config/exposure/default_configuration.json")
ALGORITHM_NAME_FIELD = "algorithm"
RUN_INTERVAL_FIELD = "run_interval"


class ExposureCompensationRunner(AlgorithmRunner):
    name = "exposure_compensation"
    repeat = True
    delay = 0.6

    def __init__(self, config_path=ALGORITHM_CONFIGURATION):
        self.config = PTV.from_file(config_path)
        super(ExposureCompensationRunner, self).__init__(str(self.config[ALGORITHM_NAME_FIELD]))
        if self.config[RUN_INTERVAL_FIELD] is not None:
            self.delay = self.config[RUN_INTERVAL_FIELD]
