#!/usr/bin/env python
import logging
import os
import shutil
import psutil

from distutils.dir_util import copy_tree
from os import path as osp

import errors
import utils.filesystem
import utils.conversions
import utils.performance
from deprecated import nginx

try:
    from pynvml import *
    nvmlPY = "OK"
except ImportError:
    nvmlPY = None

SDCARD_DEV = osp.join('/', 'dev', 'mmcblk0p1')


class SystemMonitor(object):
    @staticmethod
    def _format_bytes_stats(obj):
        r = {}
        for key in obj._fields:
            value = getattr(obj, key)
            if 'bytes' in key:
                value = utils.conversions.bytes2human(value)
            r[key] = value
        return r

    @staticmethod
    def _format_label(partition):
        mount_point_media = '/media/videostitch/'
        sdcard_name = '(SD Card)'
        label = partition.mountpoint.replace(mount_point_media, "")
        if partition.device == SDCARD_DEV:
            label += sdcard_name
        return label

    def status(self):
        """General status of the system components.
        """
        r = {
            'cpu': utils.performance.getCpuInfo(),
            'cuda': utils.performance.getCudaInfo(),
            'memory': utils.performance.getMemoryInfo(),
            'storage': self.storage(),
            'rtmp_server': self.rtmp_server(),
        }
        return r

    @staticmethod
    def storage():
        """Returns the list of storage devices in the system
        """
        mountpoint_filter = {'/boot/efi', '/sys/kernel/debug/tracing'}
        drives_list = []
        for partition in psutil.disk_partitions(all=True):
            try:
                mountpoint = partition.mountpoint
                if mountpoint not in mountpoint_filter:
                    data = vars(psutil.disk_usage(mountpoint))
                    data['mountpoint'] = mountpoint
                    data['device'] = partition.device
                    data['fs'] = partition.fstype
                    data['label'] = SystemMonitor._format_label(partition)
                    drives_list.append(data)
            except Exception as error:
                logging.error(error)
        return drives_list

    @staticmethod
    def drive_info(mp):
        devices = SystemMonitor.storage()
        for device in devices:
            if device['mountpoint'] == mp:
                return device

    @staticmethod
    def external_storage():
        """Returns the information of the external storage (SDCARD)
        """
        drives_list = []
        devices = SystemMonitor.storage()
        for device in devices:
            if device['device'] == SDCARD_DEV:
                drives_list.append(device)
        return drives_list

    @staticmethod
    def get_available_percentage(mp):
        """Returns the available percentage of the drive

        Args:
            mp(string): valid mounting point.
        """
        try:
            info = vars(psutil.disk_usage(mp))
        except:
            raise errors.InvalidParameters('Mount point name is not correct')
        info = SystemMonitor._format_memory_stats(info)
        return info

    @staticmethod
    def rtmp_server():
        obj = {
            'rtmp_server': nginx.instance.status(),
            'rtmp_recorder': nginx.recorder.status()
        }
        return obj

    @staticmethod
    def copy_log(log_path, destination_path):
        """Copies the log into the destination path

        Args:
            log_path(string): a valid log path directory.
            destination_path(string): a valid destination.

        Note:
            The destination folder will be removed if it exists and created if
            doesn't
        """
        dest_path = osp.join(destination_path, 'logs')
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        os.makedirs(dest_path)
        copy_tree(log_path, dest_path)


MONITOR = SystemMonitor()
