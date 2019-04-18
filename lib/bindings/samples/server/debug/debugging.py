import os
from os import path as osp
from blinker import signal

import defaults
import logging
import utils
from utils.settings_manager import SETTINGS
from system.drive_manager import DRIVE_MANAGER
from tornado.concurrent import run_on_executor
from API import stitcher_api

class Debugging(object):
    stitcher_executor = stitcher_api.StitcherAPI.stitcher_executor

    def __init__(self, output_manager, preset_manager):
        self.output_manager = output_manager
        self.preset_manager = preset_manager

        signal("stitching_running").connect(self._start_debug_outputs)

    @run_on_executor(executor='stitcher_executor')
    def _start_record(self, sender=None):
        try:
            self.output_manager.start_output("output_recorder",self.preset)
        except Exception:
            logging.error("Error raised", exc_info=True)
            signal("start_debug_failed").send()


    def _start_debug_outputs_internal(self, sender=None):
        if SETTINGS.auto_stream:
            preset = self.preset_manager.get(defaults.LOCAL_BROADCAST_PRESET)
            self.output_manager.start_output("stream", preset)

        if SETTINGS.auto_record:
            self.preset = {'drive_path' : defaults.USER_RECORDINGS_PATH}
            DRIVE_MANAGER.add_managed_drive(defaults.USER_RECORDINGS_PATH)
            signal("device_ok{}".format(utils.filesystem.get_mount_point(defaults.USER_RECORDINGS_PATH))).connect(self._start_record)

        if SETTINGS.profiling_time:
            self.output_manager.start_output("profiling", SETTINGS.profiling_time)
            signal("profiling_stopping").connect(self._stop_debug_outputs)

    @run_on_executor(executor='stitcher_executor')
    def _start_debug_outputs(self, sender=None):
        try:
            self._start_debug_outputs_internal(sender)
        except Exception:
            logging.error("Error raised", exc_info=True)
            signal("start_debug_failed").send()

    @run_on_executor(executor='stitcher_executor')
    def _stop_debug_outputs(self, sender=None):
        self.output_manager.stop_output("stream")
        self.output_manager.stop_output("output_recorder")

