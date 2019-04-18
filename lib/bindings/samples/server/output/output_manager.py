from blinker import signal

import errors
from utils.settings_manager import SETTINGS
from utils.ptv import PTV
from stream_output import StreamOutput
from recording_output import InputStreamRecorder, OutputStreamRecorder
from portaudio_output import PaOutput

try:
    from debug.profiling_output import ProfilingOutput
except ImportError:
    ProfilingOutput = None

class OutputManager(object):
    """
    Class that handles API requests related to manipulation of the outputs
    """

    def __init__(self, stitcher):
        self.stitcher = stitcher
        self.reset()

    def resetPreservedStreams(self):
        self.preserved = filter(lambda output : output.preserved, self.outputs.values())

    def reset(self):
        self._disconnect_signals()
        self.audio_layout = SETTINGS.current_audio_layout
        self.outputs = {}

        # Streams
        self.outputs["preview"] = StreamOutput(self.stitcher, "preview", preserved=True)
        self.outputs["stream"] = StreamOutput(self.stitcher, "streaming", critical=True)

        # Recording
        self.outputs["output_recorder"] = OutputStreamRecorder(self.stitcher, "recording", critical=True)
        self.outputs["input_recorder"] = InputStreamRecorder(self.stitcher, "input_recording", critical=True)

        # Portaudio
        if not SETTINGS.disable_audio_lineout:
            self.outputs["portaudio"] = PaOutput(self.stitcher, "portaudio", preserved=True)

        # Profiling
        if ProfilingOutput is not None:
            self.outputs["profiling"] = ProfilingOutput(self.stitcher)

        self.resetPreservedStreams()
        self._connect_signals()

    def has_ongoing_critical_output(self):
        """check if a critical output is ongoing (i.e. broadcast or recording)
        """
        if filter(lambda output : output.critical and not output.is_Stopped(), self.outputs.values()):
            return True
        return False

    def stop_outputs(self, sender=None, preserve=False):
        self.resetPreservedStreams()

        for output in filter(lambda output : output.state != "Stopped", self.outputs.values()):
            output.stop()
            if preserve and output not in self.preserved:
                self.preserved.append(output)


    def start_outputs(self, sender=None):
        for output in self.preserved:
            output.setAdditionalPreset({ 'channel_layout': self.audio_layout })
            output.start(preserve=True)

    def start_output(self, name, preset=None):
        output = self.outputs.get(name, None)
        if output:
            #check another critical output is not already started
            if output.critical and not SETTINGS.multiple_outputs\
                    and filter(lambda concurrent_output: concurrent_output != output
                                                        and concurrent_output.critical
                                                        and concurrent_output.state != "Stopped",
                               self.outputs.values()):
                raise errors.MultipleOutputsForbidden("another output is already started")

            output.setAdditionalPreset({ 'channel_layout': self.audio_layout })
            output.start(preset)

    def stop_output(self, name):
        output = self.outputs.get(name, None)
        if output:
            output.stop()

    def get_output(self, name):
        return self.outputs.get(name, None)

    def _disconnect_signals(self):
        signal("stitching_failing").disconnect(self.stop_outputs)
        signal("stitching_running").disconnect(self.start_outputs)

    def _connect_signals(self):
        signal("stitching_failing").connect(self.stop_outputs)
        signal("stitching_running").connect(self.start_outputs)

    def terminate(self):
        self.preserved = None
        self.outputs = None
