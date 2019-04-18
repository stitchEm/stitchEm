import unittest
import os
from blinker import signal
from utils import flash_checker
from time import sleep

TEST_FLASH_FILE = os.path.join('.','test_data','flash')

class TestFlashChecker(unittest.TestCase):
    def setUp(self):
        self.notified = False
        signal("box_flash_detected").connect(self._flash_notify)
        self.test_checker = flash_checker.FlashChecker(TEST_FLASH_FILE)

    def tearDown(self):
        if os.path.exists(TEST_FLASH_FILE):
            os.remove(TEST_FLASH_FILE)

    def _flash_notify(self, sender):
        self.notified = True

    def test_no_flash(self):
        self.test_checker.check_flash_file()
        self.assertFalse(self.notified)
        self.assertFalse(self.test_checker.flash_state)

    def test_flash_ok_and_remove(self):
        with open(TEST_FLASH_FILE, "w") as flash_file:
            flash_file.write(flash_checker.FLASH_STATE_SUCCESSFUL)
        self.test_checker.check_flash_file()
        self.assertTrue(self.notified)
        self.assertEqual(self.test_checker.flash_state, flash_checker.FLASH_STATE_SUCCESSFUL)

        self.notified = False
        self.test_checker.remove_flash_file()
        self.assertFalse(self.notified)
        self.assertFalse(self.test_checker.flash_state)
        self.assertFalse(os.path.exists(TEST_FLASH_FILE))

    def test_flash_ko(self):
        with open(TEST_FLASH_FILE, "w") as flash_file:
            flash_file.write(flash_checker.FLASH_STATE_UNSUCCESSFUL)
        self.test_checker.check_flash_file()
        self.assertTrue(self.notified)
        self.assertEqual(self.test_checker.flash_state, flash_checker.FLASH_STATE_UNSUCCESSFUL)
