import os
import tempfile
import unittest
import shutil

import utils.filesystem
import system.diskchecker


class TestDiskChecker(unittest.TestCase):
    BASE_MOUNTPOINT = os.path.join("/", "media", "videostitch")
    TEST_DEVICE = os.path.join(tempfile.gettempdir(), "tdiskchecker")
    TEST_DEVICE_MOUNTPOINT = os.path.join(BASE_MOUNTPOINT, "tdiskchecker_mp")

    def setUp(self):
        # self.tearDown()

        utils.filesystem.create_block_device(self.TEST_DEVICE, fs_type="vfat", size=20)
        if not os.path.exists(self.TEST_DEVICE_MOUNTPOINT):
            os.mkdir(self.TEST_DEVICE_MOUNTPOINT)
        if self.TEST_DEVICE_MOUNTPOINT not in utils.filesystem.MOUNTPOINT_WHITELIST:
            utils.filesystem.MOUNTPOINT_WHITELIST.add(self.TEST_DEVICE_MOUNTPOINT)
        utils.filesystem.mount(self.TEST_DEVICE, self.TEST_DEVICE_MOUNTPOINT)
        self.diskchecker = system.diskchecker.DiskChecker(self.TEST_DEVICE_MOUNTPOINT, recording_safety_margin=15,  recording_warning_margin=18)

    def tearDown(self):
        self.diskchecker.stop()
        utils.filesystem.unmount(self.TEST_DEVICE_MOUNTPOINT)
        shutil.rmtree(self.TEST_DEVICE_MOUNTPOINT, ignore_errors=True)
        os.remove(self.TEST_DEVICE)

    def test_device_ok(self):
        self.disk_ok_received = False

        def set_received(sender=None):
            self.disk_ok_received = True

        self.diskchecker.on_disk_ok.connect(set_received)
        self.diskchecker._check_drive(self.diskchecker.monitored_drive)

        self.assertTrue(self.disk_ok_received)
        self.assertFalse(self.diskchecker.warning_disk_full)

    def test_device_removed(self):
        self.test_device_ok()
        utils.filesystem.unmount(self.TEST_DEVICE_MOUNTPOINT)

        self.disk_removed_received = False

        def set_received(sender=None):
            self.disk_removed_received = True

        self.diskchecker.on_disk_removed.connect(set_received)
        self.diskchecker._check_drive(self.diskchecker.monitored_drive)
        self.assertTrue(self.disk_removed_received)

    def test_device_memory_warning(self):
        utils.filesystem.create_file(os.path.join(self.TEST_DEVICE_MOUNTPOINT, "tfile"), 3)

        self.disk_full_received = False

        def set_received(sender=None):
            self.disk_full_received = True

        self.diskchecker.on_disk_full.connect(set_received)
        self.diskchecker._check_drive(self.diskchecker.monitored_drive)

        self.assertFalse(self.disk_full_received)
        self.assertTrue(self.diskchecker.warning_disk_full)

    def test_device_full(self):
        utils.filesystem.create_file(os.path.join(self.TEST_DEVICE_MOUNTPOINT, "tfile"), 6)

        self.disk_full_received = False

        def set_received(sender=None):
            self.disk_full_received = True

        self.diskchecker.on_disk_full.connect(set_received)
        self.diskchecker._check_drive(self.diskchecker.monitored_drive)

        self.assertTrue(self.disk_full_received)
        self.assertTrue(self.diskchecker.warning_disk_full)
