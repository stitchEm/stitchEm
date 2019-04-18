import unittest

from algorithm.algorithm_runner import AlgorithmRunner
from project_manager import ProjectManager
from algorithm.algorithm_scheduler import AlgorithmScheduler
from utils.settings_manager import SETTINGS

TEST_PTV_FILEPATH = "./test_data/procedural-4K-nvenc.ptv"


class TestAlgorithmScheduler(unittest.TestCase):
    def setUp(self):
        SETTINGS.current_audio_source = ProjectManager.AUDIO_NOAUDIO
        self.project_manager = ProjectManager(TEST_PTV_FILEPATH)
        self.stitcher_controller = self.project_manager.create_controller()
        self.algorithm = AlgorithmRunner("testAlgo")
        self.scheduler = AlgorithmScheduler()

    def test_get_runner_no_scheduled(self):
        """
        Test that when no algorithm scheduled - None is returned
        """
        self.assertIsNone(self.scheduler.get_next_algorithm())

    def test_get_algo_when_scheduled(self):
        self.scheduler.schedule(self.algorithm)
        algorithm = self.scheduler.get_next_algorithm()
        self.assertEqual(algorithm, self.algorithm)

    # def test_get_algo_when_scheduled_with_delay(self):
    #     self.scheduler.schedule(self.algorithm, 0.001)
    #     time.sleep(0.1)
    #     algorithm = self.scheduler.get_next_algorithm()
    #     self.assertEqual(algorithm, self.algorithm)
    # As tornado is not running delay won't work -_-

    def test_unique_schedule(self):
        self.scheduler.schedule(self.algorithm)
        self.scheduler.schedule(self.algorithm)
        algorithm = self.scheduler.get_next_algorithm()
        self.assertEqual(algorithm, self.algorithm)
        self.assertIsNone(self.scheduler.get_next_algorithm())

    def test_reschedule(self):
        self.algorithm.repeat = True
        self.scheduler.reschedule(self.algorithm)
        algorithm = self.scheduler.get_next_algorithm()
        self.assertEqual(algorithm, self.algorithm)
