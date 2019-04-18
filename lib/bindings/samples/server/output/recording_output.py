import os
import time
from os import path as osp

import errors
import utils.filesystem
import subprocess
import logging
import shutil
import glob

from blinker import signal
from clientmessenger import CLIENT_MESSENGER
from deprecated.nginx import RTMPServer
from system import drive_manager
from output import WriterOutput
from utils.settings_manager import SETTINGS
from defaults import NGINX_RECORDINGS_PATH
from video_modes import VIDEO_MODES
from utils.ptv import PTV

REC_PATH_LINK = NGINX_RECORDINGS_PATH

class RecordingOutput(WriterOutput):
    """ Handle recorder state machine
    """

    def __init__(self, stitcher, name, critical=False, preserved=False):
        super(RecordingOutput, self).__init__(stitcher, name, critical, preserved)
        if not SETTINGS.disable_nginx_recording:
            self.rtmp_recorder = RTMPServer()

    def _start(self, preset, preserve):
        # Todo: parameter here is called preset to make it consistent with
        # base class interface. Now there is not preset and what we really
        # expect here is path for the place where to store recording result.
        # But situation here need to be clarified.
        if preset and "drive_path" in preset:
            self.drive = preset.get("drive_path")

        self._check_path(self.drive)
        self._subscribe_to_events()
        self._start_recording(self.drive, preset)
        self.timer.start()

    def _stop(self):
        self._stop_recording()
        self._unsubscribe_to_events()
        self.timer.reset()

    # Tools

    def _check_path(self, path):
        """Check if the destination path exists and we are allowed to write there.
        Args:
            path(string): A recording destination path.
        """
        self.mountpoint = utils.filesystem.get_mount_point(self.drive)
        self.drive_monitor = drive_manager.DRIVE_MANAGER.get_drive_monitor_for_path(self.drive)

        if not self.drive_monitor or not self.drive_monitor.is_Ok():
            raise errors.RecordingError("Can't start recording with given path {}.".format(path))

        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                raise errors.RecordingError("Can't start recording with given path {}.".format(path))

    def _stop_slot(self, sender=None):
        self.stop()

    def _subscribe_to_events(self):
        signal("no_device_detected{}".format(self.mountpoint)).connect(self._stop_slot)
        signal("invalid_device{}".format(self.mountpoint)).connect(self._stop_slot)
        signal("not_enough_memory{}".format(self.mountpoint)).connect(self._stop_slot)

    def _unsubscribe_to_events(self):
        signal("no_device_detected{}".format(self.mountpoint)).disconnect(self._stop_slot)
        signal("invalid_device{}".format(self.mountpoint)).disconnect(self._stop_slot)
        signal("not_enough_memory{}".format(self.mountpoint)).disconnect(self._stop_slot)

class InputStreamRecorder(RecordingOutput):
    """ Record input streams
    """

    def __init__(self, stitcher, name, critical=False, preserved=False):
        super(InputStreamRecorder, self).__init__(stitcher, name, critical, preserved)

    def _create_record_link(self, path):
        """Creates a symbolic link from 'rec' to any other destination
        Note:
            This is for managing the NGINX recording destination. i.e. inputs
        Args:
            path(string): A destination path.
        """
        if os.path.exists(REC_PATH_LINK):
            try:
                 os.remove(osp.join(REC_PATH_LINK))
            except OSError:
                 shutil.rmtree(osp.join(REC_PATH_LINK))
        os.symlink(path, REC_PATH_LINK)

    def _start_recording(self, drive, preset=None):
        """Uses nginx to record the input
        """
        self._create_record_link(drive)
        if SETTINGS.disable_nginx_recording:
            if not hasattr(self, "ptv"):
                self.ptv = PTV.from_string("{}")
            self.ptv["target_dir"] = drive
            self.stitcher.project_manager.controller.addSink(self.ptv.to_config())
        else:
            self.rtmp_recorder.start_recording(drive)
        # we have to manually advance the state to Started and send connected event,
        #  as there is no lib writer event in this scenario
        self.t_writer_ok()
        CLIENT_MESSENGER.send_event("output_connected", payload={'writer': self.name})

    def _stop_recording(self):
        """Stop the recording process (inputs) using nginx
        """
        if SETTINGS.disable_nginx_recording:
            self.stitcher.project_manager.controller.removeSink()
        else:
            self.rtmp_recorder.stop_recording()
        self.t_writer_completed()


class OutputStreamRecorder(RecordingOutput):
    """ Record output stream
    """

    def __init__(self, stitcher, name, critical=False, preserved=False):
        super(OutputStreamRecorder, self).__init__(stitcher, name, critical, preserved)

    def _start_recording(self, drive, preset):
        """Uses libvideostitch to record the output
        """
        self._load_preset(preset)

        file_index = int(SETTINGS.output_file_index) + 1

        #timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        while file_index != int(SETTINGS.output_file_index):
            if file_index > 99999:
               logging.info("Max number of files reached {}, resetting".format(file_index))
               file_index = 1
            path = osp.join(str(drive), self.ptv["filename"] + '{:05d}'.format(file_index))
            if not osp.isfile("{}.{}".format(path, self.ptv["type"])):
                chunk = glob.glob('{}-[0-9]*.{}'.format(path, self.ptv["type"]))
                if not chunk:
                    break
            file_index += 1

        if file_index == int(SETTINGS.output_file_index):
            logging.info("Max number of files reached {}".format(file_index))
            self.drive_monitor.disk_checker.on_disk_full.send(self)
            self.t_writer_critical_error()
            return

        SETTINGS.output_file_index = file_index

        # format /DRIVE_LOCATION/filenameXXXX.extension
        self.ptv["filename"] = path

        #If there is not bitrate defined in the preset, use the default one corresponding to the video mode
        if not self.ptv.has_key("bitrate"):
            self.ptv["bitrate"] = VIDEO_MODES[self.stitcher.project_manager.resolution]["recording"]["bitrate"] * 1024

        if (not self.ptv.has_key("channel_map")) and self.ptv["channel_layout"] == "amb_wxyz":
            # Internally our channel order is wxyz and youtube expects wyzx
            # So we set the channel map to fit youtube spec
            self.ptv["channel_map"] = [0, 3, 1, 2]

        self._add_writer()

    def _stop_recording(self):
        """Stop the recording process (output) using libvideostitch
        """
        self._remove_writer(self.writer_name)
        try: 
             subprocess.check_output(['sync', '-f', self.mountpoint], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError, e:
             logging.error("Error while recording stops : {}".format(e.output))

        self.t_writer_completed()

