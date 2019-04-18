import os.path

import errors
import vs

from algorithm.algorithm_runner import AlgorithmRunner
from clientmessenger import CLIENT_MESSENGER
from utils.ptv import PTV

CURRENT_DIR = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))

FACTORY_PANORAMA_PATH = os.path.join(CURRENT_DIR, "../config/calibration/4i_default_pano_definition.json")

class CalibrationRunner(AlgorithmRunner):
    name = "calibration"

    def __init__(self, algorithm_config, incremental=False, reset=False):
        super(CalibrationRunner, self).__init__("calibration")

        config = PTV.from_string(algorithm_config)["algorithms"][0]["config"]
        # add list_frames parameter otherwise algorithm will fail
        if "list_frames" not in config:
            config["list_frames"] = [0];
        self.config = PTV(config)

        self.reset = reset
        self.incremental = incremental

        # Todo there is a way to cancel calibration algorithm, but we don't have access to that functionality
        # through Online algorithm interface yet. See VideoStitch::Util::Algorithm::ProgressReporter and
        # CalibrationProgress classes

    def _update_config(self, panorama):
        self.config["improve_mode"] = self.incremental
        if self.incremental:
            self.config["calibration_control_points"] = \
                PTV.from_ptv_value(panorama.getControlPointListDef().serialize()).data

    def get_algo_output(self, panorama):
        if self.reset:
            ptv_parser = vs.Parser_create()
            if not ptv_parser.parse(FACTORY_PANORAMA_PATH):
                CLIENT_MESSENGER.send_error(
                    errors.AlgorithmRunningError("Parsing: Could not parse default factory pano definition"))
                return None
            pano = vs.PanoDefinition_create(ptv_parser.getRoot())
            if not pano:
                CLIENT_MESSENGER.send_error(
                    errors.AlgorithmRunningError("Cannot create pano definition"))
                return None

            pano.thisown = 1  # Acquire ownership.
            self.onPanorama(pano)
            return None

        return super(CalibrationRunner, self).get_algo_output(panorama)
