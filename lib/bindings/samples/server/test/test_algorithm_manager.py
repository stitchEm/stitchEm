import unittest

import vs
from algorithm.algorithm_manager import AlgorithmManager
from algorithm.calibration import CalibrationRunner
from project_manager import ProjectManager
from utils.settings_manager import SETTINGS

TEST_RIG_PARAMETERS_FILEPATH = "./test_data/calibration_rig_parameters.json"
TEST_PTV_FILEPATH = "./test_data/procedural-4K-nvenc.ptv"


class TestAlgorithmManager(unittest.TestCase):
    def setUp(self):
        SETTINGS.current_audio_source = ProjectManager.AUDIO_NOAUDIO
        self.project_manager = ProjectManager(TEST_PTV_FILEPATH)
        self.stitcher_controller = self.project_manager.create_controller()
        self.manager = AlgorithmManager(self.stitcher_controller)
        with open(TEST_RIG_PARAMETERS_FILEPATH, "r") as rig_params_file:
            self.rig_parameters = rig_params_file.read()

    def test_create(self):
        self.assertIsInstance(AlgorithmManager.create_algorithm(CalibrationRunner.name, self.rig_parameters), CalibrationRunner)

    def test_algorithm_running(self):
        self.manager.start_calibration(self.rig_parameters)
        algorithm = self.manager.algorithm_scheduler.scheduled_algorithms[CalibrationRunner.name]
        self.manager.get_next_algorithm(self.project_manager.panorama)
        self.assertIn(algorithm, self.manager.running_algorithms)

    def test_cancel(self):
        self.manager.start_calibration(self.rig_parameters)
        algorithm = self.manager.algorithm_scheduler.scheduled_algorithms[CalibrationRunner.name]
        self.manager.get_next_algorithm(self.project_manager.panorama)
        self.manager.cancel_running_algorithms()
        self.assertFalse(self.manager.running_algorithms)
