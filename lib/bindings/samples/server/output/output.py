from os import path as osp
import gc

import defaults
import vs
import utils.async
from transitions import MachineError
from transitions.extensions import LockedMachine as Machine
from blinker import signal

from clientmessenger import CLIENT_MESSENGER
from utils.ptv import PTV
from utils.settings_manager import SETTINGS
from utils.timer import Timer
from utils.cpp_callback import CppCallback

import errors


class Output(object):
    def __init__(self, stitcher, name, critical=False, preserved=False):

        self.stitcher = stitcher
        self.timer = Timer(name)
        self.name = name
        self.critical = critical
        self.preserved = preserved
        self.shared_video = None
        self.additional_preset = {}

        # manage if we need to fire a disconnection event
        self.event_send_disconnection = False

        # manage if we need to fire a connection event
        self.event_send_connection = True

        # State Machine

        states = ["Stopped", "Starting", "Started", "Retrying", "Stopping"]

        transitions = [

            {"trigger": "t_start",
             "prepare": "_transition_check",
             "before": "_start",
             "source": "Stopped",
             "dest": "Starting"},

            {"trigger": "t_writer_ok",
             "prepare": "_transition_check",
             "source": ["Starting", "Retrying"],
             "dest": "Started"},

            {"trigger": "t_writer_critical_error",
             "prepare": "_transition_check",
             "before": "_stop",
             "source": ["Starting", "Retrying", "Started"],
             "dest": "Stopping"},

            {"trigger": "t_writer_recoverable_error",
             "prepare": "_transition_check",
             "source": ["Starting", "Retrying", "Started"],
             "dest": "Retrying"},

            {"trigger": "t_stop",
             "prepare": "_transition_check",
             "before": "_stop",
             "source": ["Starting", "Retrying", "Started"],
             "dest": "Stopping"},

            {"trigger": "t_writer_completed",
             "prepare": "_transition_check",
             "source": "Stopping",
             "dest": "Stopped"},

            # avoid exception if ever a connection error arrives after we have stopped retrying
            {"trigger": "t_writer_ok",
             "source": "Stopped",
             "dest": "Stopped"},
            {"trigger": "t_writer_critical_error",
             "source": "Stopped",
             "dest": "Stopped"},
            {"trigger": "t_writer_recoverable_error",
             "source": "Stopped",
             "dest": "Stopped"},
            {"trigger": "t_writer_ok",
             "source": "Started",
             "dest": "Started"},

        ]

        self.machine = Machine(
            name=self.name, model=self, states=states,
            transitions=transitions, initial='Stopped', async=True)

    def setAdditionalPreset(self, additional_preset):
        self.additional_preset = additional_preset

    def start(self, preset=None, preserve=False):
        """Start Output"""
        try:
            self.t_start(preset, preserve)
        except MachineError:
            raise errors.OutputAlreadyStarted(self.name)

    def stop(self):
        """Stop Output"""
        try:
            self.t_stop()
        except MachineError:
            # Stopping twice is not considered as an error
            pass

    def get_statistics(self):
        """
        :return: Retrieve output statistics
        """

    def _start(self, *args, **kwargs):
        """Internal implementation of start"""

    def _stop(self):
        """Internal implementation of stop"""

    def _transition_check(self, *args, **kwargs):
        if self.stitcher.state != "Running":
            raise errors.StitcherNotStarted(self.name)

    # manage disconnection vs not able to connect
    def on_enter_Started(self):
        self.event_send_disconnection = True
        self.event_send_connection = False
        signal("output_started_{}".format(self.name)).send()

    def on_enter_Stopped(self):
        self.event_send_disconnection = False
        signal("output_stopped_{}".format(self.name)).send()

    def on_enter_Retrying(self):
        self.event_send_disconnection = False
        signal("output_stopped_{}".format(self.name)).send()


class WriterOutput(Output):
    """ Base class for writer based outputs
    """

    def __init__(self, stitcher, name, critical=False, preserved=False):
        super(WriterOutput, self).__init__(stitcher, name, critical, preserved)
        self.callbacks = []
        self.preset = None

    def _load_preset(self, preset=None, preserve=False):
        """Creates configuration object based on the default preset and the given one if present

        Args:
            preset:
        """
        if SETTINGS.ptv is not None:
            preset_ptv = PTV.from_file(SETTINGS.ptv)
            self.ptv = PTV(preset_ptv[self.name])
        else:
            self.ptv = PTV.from_file(self._get_preset_filepath(defaults.SYSTEM_PRESETS_DIR_PATH,
                                                               defaults.DEFAULT_PRESET_FILENAME_NOEXT))

            if isinstance(preset, str) or isinstance(preset, unicode):
                preset_ptv = PTV.from_file(self._get_preset_filepath(defaults.USER_PRESETS_DIR_PATH,
                                                                      preset))
            elif preset is not None:
                preset_ptv = PTV(preset)
            else:
                preset_ptv = None

            if preset_ptv:
                if preserve and self.preset:
                    self.preset.merge(preset_ptv)
                else:
                    self.preset = preset_ptv

            if self.preset:
                self.ptv.merge(self.preset)

            if self.additional_preset:
                self.ptv.merge(self.additional_preset)

            if self.ptv["channel_layout"] == "amb_wxyz":
                self.ptv["audio_bitrate"] = self.ptv["ambisonic_audio_bitrate"]

    def _get_preset_filepath(self, preset_dir, preset_name):
        return osp.join(preset_dir, self.name, preset_name + ".preset")

    def _add_writer(self):

        if "filename" not in self.ptv:
            raise errors.OutputError('Configuration preset has no filename field')

        self.writer_name = str(self.ptv["filename"])
        self.config = self.ptv.to_config()
        writer = self._create_writer(self.stitcher.project_manager.panorama, self.stitcher.project_manager.controller, self.config, self.writer_name)

        self.shared_writer = vs.writerSharedPtr(writer.release())
        self.connect_writer_events(self.shared_writer)
        self.shared_video = vs.videoWriterSharedPtr(self.shared_writer)
        shared_audio = vs.audioWriterSharedPtr(self.shared_writer)
        self.has_audio = shared_audio is not None
        if self.shared_video is not None and not self.stitcher.stitch_output.addWriter(self.shared_video):
            raise errors.InternalError("Cannot add video writer to stitcher")
        if shared_audio is not None and not self.stitcher.project_manager.controller.addAudioOutput(shared_audio):
            raise errors.InternalError("Cannot add audio writer to stitcher")

        self.timer.start()
        gc.collect()

    def get_preset_name(self):
        return self.preset.get("name") if self.preset else None

    def connect_writer_events(self, shared_writer):
        """
        Add reaction for the events from plugin
        :param shared_writer:
        :return:
        """

        def connect(event, callback):
            self.callbacks.append(CppCallback(callback))
            shared_writer.getOutputEventManager().subscribe(event, self.callbacks[-1].toFunction())

        # we don't care about obvious connecting event

        # connected event treatment
        def connected_callback(payload_message):
            if self.state != "Started" or self.event_send_connection == True:
                self.event_send_connection = False
                utils.async.defer(self.t_writer_ok)
                CLIENT_MESSENGER.send_event("output_connected", payload={'writer': self.name})

        connect(vs.OutputEventManager.EventType_Connected, connected_callback)

        # disconnected event treatment
        def disconnected_callback(payload_message):
            self.event_send_connection = True
            if self.event_send_disconnection:
                event_context = "disconnected"
            else:
                event_context = "cannot_connect"
            if self._writer_error_is_recoverable(payload_message):
                utils.async.defer(self.t_writer_recoverable_error)
                event_name = "output_{}_retrying"
            else:
                utils.async.defer(self.t_writer_critical_error)
                event_name = "output_{}_stopping"
            CLIENT_MESSENGER.send_event(event_name.format(event_context),
                                        payload={'writer': self.name, 'code': payload_message})

        connect(vs.OutputEventManager.EventType_Disconnected, disconnected_callback)

    def flush_writer_events(self):
        """
        Make sure events from plugin are treated
        :return:
        """
        for callback in self.callbacks:
            callback.join()
        self.callbacks = []


    def _writer_error_is_recoverable(self, payload):
        """
        check if we restart the output or we stop here
        default behavior is stopping
        """
        return False

    def _remove_writer(self, writer_name):
        """Stops the stream
        """
        if self.has_audio:
            success = self.stitcher.project_manager.controller.removeAudioOutput(writer_name)
            if not success:
                raise errors.InternalError("Cannot remove audio writer {}".format(writer_name))
        success = self.stitcher.stitch_output.removeWriterNoGIL(writer_name)
        if not success:
            raise errors.InternalError("Cannot remove video writer")

        self.shared_writer = None
        self.shared_video = None
        self.timer.reset()
        gc.collect()
        self.flush_writer_events()

    def _create_writer(self, panorama, controller, config, writer_name):
        """Creates an output writer to register to the stitcher
        """
        sampling_rate = self.ptv.get("sampling_rate", 0)
        sample_format = str(self.ptv.get("sample_format", ""))
        channel_layout = str(self.ptv.get("channel_layout", ""))

        writer = vs.Output_createNoGIL(
            config,
            writer_name,
            panorama.width,
            panorama.height,
            controller.getFrameRateFromInputController(),
            sampling_rate,
            sample_format,
            channel_layout
        )

        if not writer.status().ok():
            raise errors.InternalError(
                "Cannot create writer for output {}: {}, {}".format(
                    self.writer_name,
                    str(writer.status().getOrigin()),
                    writer.status().getErrorMessage()))

        return writer

    def _get_latency(self):
        if self.shared_video is not None:
            return self.shared_video.getLatency()
        else:
            return -1
