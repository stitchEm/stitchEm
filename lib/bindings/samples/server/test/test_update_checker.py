import tempfile
import unittest
import os
from time import sleep
from blinker import signal
import utils.update_checker
import API.version
import logging

REACH_UPDATE_LOCAL_FILE_MAX_DELAY = 0.2  # in seconds
DESTROY_FILE_DELAY = 0.2
LOCAL_UPDATE_FILE_CONTENT = "{ \"version\": \"1.2.3\", \"release notes\": \"<p>hello world</p>\" }"
LOCAL_UPDATE_FILE_CONTENT_ALT = "{ \"version\": \"1.2.4\", \"release notes\": \"<p>hello world</p>\" }"
LOCAL_UPDATE_FILE_NAME = "local_update.release"
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),"test_data")

# reduce the time before checking again
utils.update_checker.UPDATE_CHECK_INTERVAL_SHORT = 0.1
utils.update_checker.UPDATE_CHECK_INTERVAL_LONG = 0.1


class TestUpdateChecker(unittest.TestCase):
    def setUp(self):
        self.notified = False
        signal("update_available").connect(self._update_notify)
        self.local_update_path = os.path.join(TEST_DATA_DIR, LOCAL_UPDATE_FILE_NAME)
        self.local_update_url = "file://{}".format(self.local_update_path)
        with open(self.local_update_path, 'w') as local_file:
            local_file.write(LOCAL_UPDATE_FILE_CONTENT)

    def tearDown(self):
        # remove local file
        os.remove(self.local_update_path)
        sleep(DESTROY_FILE_DELAY)

    def _update_notify(self, sender):
        self.notified = True

    def test_check_update_available(self):
        API.version.SERVER_VERSION = "1.2.2"
        update_checker = utils.update_checker.UpdateChecker(self.local_update_url)
        sleep(REACH_UPDATE_LOCAL_FILE_MAX_DELAY)
        self.assertTrue(update_checker.update_info)
        self.assertEqual(update_checker.update_info[utils.update_checker.UPDATE_INFO_VERSION_KEY],
                         "1.2.3")
        self.assertEqual(update_checker.update_info[utils.update_checker.UPDATE_INFO_RELEASE_NOTES_KEY],
                         "<p>hello world</p>")
        self.assertTrue(self.notified)
        update_checker.stop()

    def test_check_update_notification(self):
        API.version.SERVER_VERSION = "1.2.3"
        update_checker = utils.update_checker.UpdateChecker(self.local_update_url)
        sleep(REACH_UPDATE_LOCAL_FILE_MAX_DELAY)
        self.assertFalse(update_checker.update_info)
        self.assertFalse(self.notified)
        # change file
        with open(self.local_update_path, 'w') as local_file:
            local_file.write(LOCAL_UPDATE_FILE_CONTENT_ALT)
        #wait so the file is reachable through url
        sleep(REACH_UPDATE_LOCAL_FILE_MAX_DELAY)
        #wait to be sure the modified file has been checked
        sleep(utils.update_checker.UPDATE_CHECK_INTERVAL_LONG)
        self.assertTrue(update_checker.update_info)
        self.assertTrue(self.notified)
        update_checker.stop()

    def test_check_no_update_available(self):
        API.version.SERVER_VERSION = "1.3.0"
        update_checker = utils.update_checker.UpdateChecker(self.local_update_url)
        sleep(REACH_UPDATE_LOCAL_FILE_MAX_DELAY)
        self.assertFalse(update_checker.update_info)
        self.assertFalse(self.notified)
        update_checker.stop()
