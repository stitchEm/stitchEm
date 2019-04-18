import os
import tempfile
import unittest

import time

import shutil
from blinker import signal

import utils.filesystem
from system.diskchecker import DiskChecker
from system.drive_monitor import DriveMonitor
from utils.settings_manager import SETTINGS
from defaults import DEFAULT_OPTIONS


class TestSDCardManager(unittest.TestCase):
    """deps: fusefat, fuse2fs, ntfs utils, ntfs3g"""
    LONG_SLEEP = 0.5
    SHORT_SLEEP = 0.1

    BASE_MOUNTPOINT = os.path.join("/", "media", "videostitch")
    VFAT_DEVICE = os.path.join(tempfile.gettempdir(), "vfatdev")
    VFAT_MOUNTPOINT = os.path.join(BASE_MOUNTPOINT, "vfat_mp")
    NTFS_DEVICE = os.path.join(tempfile.gettempdir(), "ntfsdev")
    NTFS_MOUNTPOINT = os.path.join(BASE_MOUNTPOINT, "ntfs_mp")
    VFAT_RO_DEVICE = os.path.join(tempfile.gettempdir(), "vfatrodev")
    VFAT_RO_MOUNTPOINT = os.path.join(BASE_MOUNTPOINT, "vfatro_mp")

    def setUp(self):
        # self.tearDown()

        SETTINGS.recording_safety_margin = 15

        utils.filesystem.create_block_device(self.VFAT_DEVICE, "vfat")
        utils.filesystem.create_block_device(self.VFAT_RO_DEVICE, "vfat")
        # TODO: have ntfs on buildbot behave the same way it behaves on SB image
        # then "ntfs" could be used instead of "exfat" for invalid device"
#        utils.filesystem.create_block_device(self.NTFS_DEVICE, "ntfs", additional_params=["-F"])
        utils.filesystem.create_block_device(self.NTFS_DEVICE, "exfat")

        if not os.path.exists(self.VFAT_MOUNTPOINT):
            os.mkdir(self.VFAT_MOUNTPOINT)
        if not os.path.exists(self.VFAT_RO_MOUNTPOINT):
            os.mkdir(self.VFAT_RO_MOUNTPOINT)
        if not os.path.exists(self.NTFS_MOUNTPOINT):
            os.mkdir(self.NTFS_MOUNTPOINT)

        if self.VFAT_MOUNTPOINT not in utils.filesystem.MOUNTPOINT_WHITELIST:
            utils.filesystem.MOUNTPOINT_WHITELIST.add(self.VFAT_MOUNTPOINT)
        if self.VFAT_RO_MOUNTPOINT not in utils.filesystem.MOUNTPOINT_WHITELIST:
            utils.filesystem.MOUNTPOINT_WHITELIST.add(self.VFAT_RO_MOUNTPOINT)
        if self.NTFS_MOUNTPOINT not in utils.filesystem.MOUNTPOINT_WHITELIST:
            utils.filesystem.MOUNTPOINT_WHITELIST.add(self.NTFS_MOUNTPOINT)

        # reduce time spent on test
        DriveMonitor.CHECK_INTERVAL = 0.01
        DiskChecker.CHECK_INTERVAL = 0.01

    def tearDown(self):
        #restore defaults
        SETTINGS.recording_safety_margin = DEFAULT_OPTIONS["recording_safety_margin"]

        #leave a bit of time for DriveMonitor to be properly destroyed
        time.sleep(self.SHORT_SLEEP)
        """umount, remove devices and mountpoints"""
        utils.filesystem.unmount(self.VFAT_MOUNTPOINT)
        utils.filesystem.unmount(self.NTFS_MOUNTPOINT)
        utils.filesystem.unmount(self.VFAT_RO_MOUNTPOINT)

        shutil.rmtree(self.VFAT_MOUNTPOINT, ignore_errors=True)
        shutil.rmtree(self.NTFS_MOUNTPOINT, ignore_errors=True)
        shutil.rmtree(self.VFAT_RO_MOUNTPOINT, ignore_errors=True)

        os.remove(self.VFAT_DEVICE)
        os.remove(self.NTFS_DEVICE)
        if os.path.exists(self.VFAT_RO_DEVICE):
            os.remove(self.VFAT_RO_DEVICE)

    def test_nodevice(self):
        test_manager = DriveMonitor("/tmp/tmountpoint", "/tmp/nodev")
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_NoDeviceDetected())
        test_manager.stop()

    def test_invalid_device(self):
        test_manager = DriveMonitor("/tmp/tmountpoint", self.NTFS_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_InvalidDevice(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_device_not_compatible(self):
        utils.filesystem.mount(self.NTFS_DEVICE, self.NTFS_MOUNTPOINT)
        test_manager = DriveMonitor(self.NTFS_MOUNTPOINT, self.NTFS_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceNotCompatible(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_device_ok(self):
        utils.filesystem.mount(self.VFAT_DEVICE, self.VFAT_MOUNTPOINT)
        test_manager = DriveMonitor(self.VFAT_MOUNTPOINT, self.VFAT_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceOk(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_no_space_left(self):
        utils.filesystem.mount(self.VFAT_DEVICE, self.VFAT_MOUNTPOINT)
        utils.filesystem.create_file(os.path.join(self.VFAT_MOUNTPOINT, "tfile"), 5)
        test_manager = DriveMonitor(self.VFAT_MOUNTPOINT, self.VFAT_DEVICE)
        time.sleep(self.LONG_SLEEP)
        self.assertTrue(test_manager.is_NotEnoughMemory(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_remove_device(self):
        utils.filesystem.mount(self.VFAT_DEVICE, self.VFAT_MOUNTPOINT)
        test_manager = DriveMonitor(self.VFAT_MOUNTPOINT, self.VFAT_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceOk(), msg="Current state: {}".format(test_manager.state))
        utils.filesystem.unmount(self.VFAT_MOUNTPOINT)
        time.sleep(self.LONG_SLEEP)
        self.assertTrue(test_manager.is_InvalidDevice(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def format_invalid(self, test_manager):
        test_manager.format_drive()
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceOk(), msg="Current state: {}".format(test_manager.state))
        utils.filesystem.create_file(os.path.join(self.NTFS_MOUNTPOINT,"writetest"), size=1)
        test_manager.stop()

    def test_format_invalid_unmounted(self):
        test_manager = DriveMonitor(self.NTFS_MOUNTPOINT, self.NTFS_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_InvalidDevice(), msg="Current state: {}".format(test_manager.state))
        self.format_invalid(test_manager)
        test_manager.stop()

    def test_format_notcompatible_mounted(self):
        utils.filesystem.mount(self.NTFS_DEVICE, self.NTFS_MOUNTPOINT)
        test_manager = DriveMonitor(self.NTFS_MOUNTPOINT, self.NTFS_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceNotCompatible(), msg="Current state: {}".format(test_manager.state))
        self.format_invalid(test_manager)
        test_manager.stop()

    def test_format_exfat(self):
        test_manager = DriveMonitor(self.NTFS_MOUNTPOINT, self.NTFS_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_InvalidDevice(), msg="Current state: {}".format(test_manager.state))
        test_manager.format_drive("exfat")
        time.sleep(self.SHORT_SLEEP)
        if 'fuseblk' in test_manager.SUPPORTED_FILESYSTEMS:
          self.assertTrue(test_manager.is_DeviceOk(), msg="Current state: {}".format(test_manager.state))
        else:
          self.assertTrue(test_manager.is_DeviceNotCompatible(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_format_full(self):
        utils.filesystem.mount(self.VFAT_DEVICE, self.VFAT_MOUNTPOINT)
        utils.filesystem.create_file(os.path.join(self.VFAT_MOUNTPOINT, "tfile"), 19.9)
        test_manager = DriveMonitor(self.VFAT_MOUNTPOINT, self.VFAT_DEVICE)
        time.sleep(self.LONG_SLEEP)
        self.assertTrue(test_manager.is_NotEnoughMemory(), msg="Current state: {}".format(test_manager.state))
        test_manager.format_drive()
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceOk(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_format_then_full(self):
        utils.filesystem.mount(self.VFAT_DEVICE, self.VFAT_MOUNTPOINT)
        test_manager = DriveMonitor(self.VFAT_MOUNTPOINT, self.VFAT_DEVICE)
        time.sleep(self.LONG_SLEEP)
        self.assertTrue(test_manager.is_Ok(), msg="Current state: {}".format(test_manager.state))
        test_manager.format_drive()
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_Ok(), msg="Current state: {}".format(test_manager.state))
        try:
            utils.filesystem.create_file(os.path.join(self.VFAT_MOUNTPOINT, "tfile"), 19.9)
        except:
            pass
        time.sleep(self.LONG_SLEEP)
        self.assertTrue(test_manager.is_NotEnoughMemory(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_format_nodev(self):
        test_manager = DriveMonitor("/tmp/tmountpoint", "/tmp/nodev")
        self.assertTrue(test_manager.is_NoDeviceDetected())
        self.assertRaises(IOError, test_manager.format_drive, SETTINGS.sdcard_filesystem)
        test_manager.stop()

    def warn_cluster(self, sender):
        self.cluster_size = sender.cluster_size

    def test_small_cluster_size(self):
        utils.filesystem.create_block_device(self.VFAT_DEVICE, "vfat",
                                             additional_params=["-F", "32", "-s", "1", "-S", "512"])
        self.cluster_size = 0
        signal("device_warn_cluster{}".format(self.VFAT_MOUNTPOINT)).connect(self.warn_cluster)
        test_manager = DriveMonitor(self.VFAT_MOUNTPOINT, self.VFAT_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        utils.filesystem.mount(self.VFAT_DEVICE, self.VFAT_MOUNTPOINT)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceOk(), msg="Current state: {}".format(test_manager.state))
        self.assertTrue((self.cluster_size == 512),
                        msg="Small Cluster size (512) should be detected: {}".format(self.cluster_size))
        signal("device_warn_cluster{}".format(self.VFAT_MOUNTPOINT)).disconnect(self.warn_cluster)
        test_manager.stop()

    def test_good_cluster_size(self):
        utils.filesystem.create_block_device(self.VFAT_DEVICE, "vfat", 
                                             additional_params=["-F", "32", "-s", "64", "-S", "512"])
        self.cluster_size = 0
        signal("device_warn_cluster{}".format(self.VFAT_MOUNTPOINT)).connect(self.warn_cluster)
        test_manager = DriveMonitor(self.VFAT_MOUNTPOINT, self.VFAT_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        utils.filesystem.mount(self.VFAT_DEVICE, self.VFAT_MOUNTPOINT)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceOk(), msg="Current state: {}".format(test_manager.state))
        self.assertTrue((self.cluster_size == 0),
                        msg="Small Cluster size should not be detected: {}".format(self.cluster_size))
        signal("device_warn_cluster{}".format(self.VFAT_MOUNTPOINT)).disconnect(self.warn_cluster)
        test_manager.stop()

    def test_read_only_fs(self):
        test_manager = DriveMonitor(self.VFAT_RO_MOUNTPOINT, self.VFAT_RO_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        utils.filesystem.mount(self.VFAT_RO_DEVICE, self.VFAT_RO_MOUNTPOINT)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceReadOnly(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_format_read_only_fs(self):
        os.chmod(self.VFAT_RO_DEVICE,0o444)
        test_manager = DriveMonitor(self.VFAT_RO_MOUNTPOINT, self.VFAT_RO_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_InvalidDevice(), msg="Current state: {}".format(test_manager.state))
        time.sleep(self.SHORT_SLEEP)
        test_manager.format_drive()
        time.sleep(self.LONG_SLEEP)
        self.assertTrue(test_manager.is_DeviceReadOnly(), msg="Current state: {}".format(test_manager.state))
        utils.filesystem.unmount(self.VFAT_RO_MOUNTPOINT)
        os.remove(self.VFAT_RO_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_NoDeviceDetected(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()

    def test_eject(self):
        utils.filesystem.mount(self.VFAT_DEVICE, self.VFAT_MOUNTPOINT)
        test_manager = DriveMonitor(self.VFAT_MOUNTPOINT, self.VFAT_DEVICE)
        time.sleep(self.SHORT_SLEEP)
        self.assertTrue(test_manager.is_DeviceOk(), msg="Current state: {}".format(test_manager.state))
        test_manager.eject_drive()
        time.sleep(self.LONG_SLEEP)
        self.assertTrue(test_manager.is_DeviceRemovable(), msg="Current state: {}".format(test_manager.state))
        test_manager.stop()


