from defaults import ENCODER_VERSION

from output import WriterOutput

STREAMING_ERROR_CONNECTION_REFUSED = "RTMPConnectionRefusedMessage"
STREAMING_ERROR_BAD_URL = "BadUrl"
STREAMING_ERROR_NETWORK_ERROR = "NetworkError"

class StreamOutput(WriterOutput):
    """ Handle stream state machine
    """

    def __init__(self, stitcher, name, critical=False, preserved=False):
        super(StreamOutput, self).__init__(stitcher, name, critical, preserved)

    def _start(self, preset, preserve):
        """Applies the streaming preset and starts the stream
        """
        self._load_preset(preset, preserve)

        self.ptv["user_agent"] = ENCODER_VERSION
        if self.ptv.has_key("bitrate") and not self.ptv.has_key("vbvMaxBitrate"):
            self.ptv["vbvMaxBitrate"] = self.ptv["bitrate"]
        if self.ptv.has_key("bitrate") and not self.ptv.has_key("bitrate_min"):
            self.ptv["bitrate_min"] = self.ptv["bitrate"] / 2
        self._add_writer()

    def _stop(self):
        """Stops the stream
        """
        self._remove_writer(self.writer_name)
        self.t_writer_completed()

    def _writer_error_is_recoverable(self, payload_message):
        """
        check if we restart the output or we stop here
        default behavior is stopping
        """
        # payload is (writer_name, error_stringcode)
        if payload_message == STREAMING_ERROR_NETWORK_ERROR:
            return True
        return False

    def get_latency(self):
        return self._get_latency()
