import json
import unittest
import os
from utils.settings_manager import SettingsManager

TEST_SETTINGS = os.path.join("test_data", "server_settings.json")


class Tobject(object):
    pass


class TestSettingsManager(unittest.TestCase):
    default_settings = {"resolution" : "4K DCI", "loglevel": 3}
    def tearDown(self):
        with open(TEST_SETTINGS, mode="w") as ts_file:
            json.dump(self.default_settings, ts_file)

    def test_read_file(self):
        tmanager = SettingsManager(object, settings_path=TEST_SETTINGS)
        self.assertEqual(tmanager.resolution, "4K DCI")
        self.assertEqual(tmanager.loglevel, 3)

    def test_read_cmdline(self):
        tobject = Tobject()
        setattr(tobject, "ta1", 12)
        setattr(tobject, "ta2", "ta2")
        tmanager = SettingsManager(tobject)
        self.assertEqual(tmanager.ta1, tobject.ta1)
        self.assertEqual(tmanager.ta2, tobject.ta2)

    def test_cmdline_overwrite(self):
        tobject = Tobject()
        setattr(tobject, "resolution", "wheee")
        setattr(tobject, "ta2", "ta2")

        tmanager = SettingsManager(tobject, settings_path=TEST_SETTINGS)

        self.assertEqual(tmanager.resolution, "wheee")
        self.assertEqual(tmanager.ta2, tobject.ta2)
        self.assertEqual(tmanager.loglevel, 3)

    def test_autosave(self):
        tmanager = SettingsManager(object, settings_path=TEST_SETTINGS)
        tmanager.ts3 = "ts3"
        tmanager.ta2 = "ta2"

        tmanager2 = SettingsManager(object, settings_path=TEST_SETTINGS)
        self.assertEqual(tmanager.ts3, tmanager2.ts3)
        self.assertEqual(tmanager.ta2, tmanager2.ta2)
        self.assertEqual(tmanager2.resolution, "4K DCI")

    def test_autosave_no_file(self):
        nonexistant_path = os.path.join("test_data", "nonexistant_file")
        tmanager = SettingsManager(object, settings_path=nonexistant_path)
        tmanager.ts3 = "ts3"

        tmanager2 = SettingsManager(object, settings_path=nonexistant_path)
        self.assertEqual(tmanager.ts3, tmanager2.ts3)

        os.remove(nonexistant_path)


    def test_errors(self):
        tmanager = SettingsManager(object, settings_path=TEST_SETTINGS)
        tmanager.ts3 = "ts3"
        tmanager.ta2 = "ta2"
        with open(TEST_SETTINGS, "a") as f:
            f.write("garbage")

        tmanager2 = SettingsManager(object, settings_path=TEST_SETTINGS)
        self.assertEqual(tmanager2.resolution, "4K DCI")


        tmanager.ta3 = "ta3"
        tmanager3 = SettingsManager(object, settings_path=TEST_SETTINGS)
        self.assertEqual(tmanager3.resolution, "4K DCI")
        self.assertEqual(tmanager.ta3, tmanager3.ta3)
