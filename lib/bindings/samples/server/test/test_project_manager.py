import unittest
import vs
from project_manager import ProjectManager
from utils.settings_manager import SETTINGS

TEST_PTV_FILEPATH = "./test_data/procedural-4K-nvenc.ptv"
TEST_PTV_SAVEPATH = "./test_data/project_manager_save.ptv"


class TestProjectManager(unittest.TestCase):
    def setUp(self):
        SETTINGS.current_audio_source = ProjectManager.AUDIO_NOAUDIO
        self.project_manager = ProjectManager(TEST_PTV_FILEPATH)

    def test_save(self):
        exposure = self.project_manager.project_config.has("exposure")
        exposure = exposure.clone()

        exposure.push("updated", vs.Value.boolObject(True))

        self.project_manager.update("exposure", exposure, auto_save=False)
        self.project_manager.save(TEST_PTV_SAVEPATH)

        updated_project = ProjectManager(TEST_PTV_SAVEPATH)

        self.assertEqual(exposure.has("updated").getBool(), updated_project.project_config.has("exposure").has("updated").getBool())
