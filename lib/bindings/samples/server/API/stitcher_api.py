from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor

from API.handlers import APIHandler
from API.schema import api
from utils.settings_manager import SETTINGS
import errors


class StitcherAPI(APIHandler):
    """REST interface to the stitcher and related actions.
    """
    executor = ThreadPoolExecutor(1)
    reset_executor = ThreadPoolExecutor(1)
    stitcher_executor = ThreadPoolExecutor(1)

    def __init__(self, extra):
        """Init
        """
        super(StitcherAPI, self).__init__(extra)
        self.server = extra["server"]
        self.stitcher = extra["video_stitcher"]
        self.output_manager = extra["output_manager"]
        self.preset_manager = extra["preset_manager"]
        self.project_manager = extra["project_manager"]

    @api(name="StartRecording",
         endpoint="stitcher.start_recording",
         description="Start recording Inputs, Output or InputsOutput in the listed drives",
         parameters={
             "type": "object",
             "properties":
                 {
                     "output_preset": {
                         "type": "object",
                         "properties": {
                             "drive_path": {
                                 "type": "string"
                             }
                         },
                         "description": "The output recording preset."},
                     "inputs_preset": {
                         "type": "object",
                         "properties": {
                             "drive_path": {
                                 "type": "string"
                             }
                         },
                         "description": "The inputs recording preset."},
                 }
         },
         errors=[errors.RecordingError, errors.StitcherError,
                 errors.CameraError]
         )
    @run_on_executor(executor='stitcher_executor')
    def start_recording(self, parameters):
        inputs_preset = parameters.get("inputs_preset")
        output_preset = parameters.get("output_preset")
        if inputs_preset:
            self.output_manager.start_output("input_recorder", inputs_preset)

        if output_preset:
            self.output_manager.start_output("output_recorder", output_preset)

    @api(name="StopRecording",
         endpoint="stitcher.stop_recording",
         description="Stop the inputs and/or output recording",
         errors=[errors.RecordingError, errors.StitcherError,
                 errors.CameraError]
         )
    @run_on_executor
    def stop_recording(self, parameters=None):
        self.output_manager.stop_output("input_recorder")
        self.output_manager.stop_output("output_recorder")

    @api(name="StartStream",
         endpoint="stitcher.start_stream",
         description="Starts the RTMP stream",
         parameters={
             "type": ["object", "null"],
             "properties":
                 {
                     "preset": {
                         "type": "string",
                         "description": "A streaming preset"},
                 },
             "required": ["preset"]
         },
         errors=[errors.PresetDoesNotExist, errors.StreamingError, errors.StitcherError]
         )
    @run_on_executor(executor='stitcher_executor')
    def start_stream(self, parameters):
        preset = self.preset_manager.get(parameters.get("preset"))
        if (preset is None):
            raise errors.PresetDoesNotExist("preset {} not found".format(parameters.get("preset")))

        self.output_manager.start_output("stream", preset)
        SETTINGS.last_preset_streaming = preset["name"]

    @api(name="StopStream",
         endpoint="stitcher.stop_stream",
         description="Stops the RTMP stream",
         errors=[errors.StreamingError, errors.StitcherError]
         )
    @run_on_executor
    def stop_stream(self, parameters=None):
        self.output_manager.stop_output("stream")

    @run_on_executor(executor='stitcher_executor')
    def start_preview(self, parameters=None):
        self.output_manager.start_output("preview", parameters)

    @run_on_executor
    def stop_preview(self, parameters=None):
        self.output_manager.stop_output("preview")

    @run_on_executor
    def get_record_config(self, parameters=None):
        """Get the current project recording configuration.

        Returns:
            - ``ConfigurationError``
            - Otherwise: the recording configuration.
        """
        from output import Output
        return Output.get_output_config(self.output_manager.record_conf)

    @api(name="GetStreamPresets",
         endpoint="stitcher.get_stream_presets",
         description="Returns a list of streaming presets",
         result={
             "type": "object",
             "properties":
                 {
                     "entries": {
                         "type": "array",
                         "items": {
                             "$ref": "StreamPreset"
                         }
                     }
                 }
         },
         errors=[errors.PresetError]
         )
    @run_on_executor
    def get_stream_presets(self, parameters=None):
        return {"entries": self.preset_manager.list()}

    @api(name="CreateStreamPreset",
         endpoint="stitcher.create_stream_preset",
         description="Create a streaming preset",
         parameters={
             "$ref": "StreamPreset"
         },
         errors=[errors.PresetError]
         )
    @run_on_executor
    def create_stream_preset(self, parameters):
        self.preset_manager.create(parameters)

    @api(name="RemoveStreamPreset",
         endpoint="stitcher.remove_stream_preset",
         description="Remove a streaming preset",
         parameters={
             "type": "object",
             "properties":
                 {
                     "name": {
                         "type": "string",
                         "description": "The streaming preset name"},
                 },
             "minProperties": 1
         },
         errors=[errors.PresetError]
         )
    @run_on_executor
    def remove_stream_preset(self, parameters):
        name = parameters.get("name")
        self.preset_manager.remove(name)

    @api(name="GetInfo",
         endpoint="stitcher.get_info",
         description="Get the current project information",
         result={
             "type": "object",
             "properties":
                 {
                     "pano_size": {"type": "string"},
                     "preview_size": {"type": "string"},
                     "inputs": {"type": "integer"},
                     "inputs_size": {"type": "string"},
                     "has_audio": {"type": "boolean"},
                 }
         }
         )
    @run_on_executor
    def get_info(self, parameters=None):
        panorama_size = self.project_manager.get_panorama_size()
        inputs_size = self.project_manager.get_input_size()
        preview_size = self.project_manager.get_preview_size()
        config = {"pano_size": "{}x{}".format(panorama_size[0], panorama_size[1]),
                  "preview_size": "{}x{}".format(preview_size[0], preview_size[1]),
                  "inputs": self.project_manager.get_num_inputs(),
                  "inputs_size": "{}x{}".format(inputs_size[0], inputs_size[1]),
                  "has_audio": bool(self.stitcher.has_audio)}
        return config

    @api(name="ListVideoModes",
         endpoint="stitcher.list_video_modes",
         description="Get the list of supported video modes and associated encoding capabilities",
         result={
             "type": "object",
             "properties":
                 {
                     "video_modes": {
                         "type": "object",
                         "patternProperties": {
                             "4K DCI|4K UHD|2.8K|2K|HD": {
                                 "type": "object",
                                 "patternProperties": {
                                     "baseline|extended|main|high": {
                                         "type": "object",
                                         "properties": {
                                             "min_bitrate": {"type": "integer"},
                                             "max_bitrate": {"type": "integer"}
                                         }
                                     }
                                 }
                             }
                         }
                     }
                 }
         }
         )
    @run_on_executor
    def list_video_modes(self, parameters=None):
        return {"video_modes": self.project_manager.get_video_modes()}

    @api(name="SetVideoMode",
         endpoint="stitcher.set_video_mode",
         description="Set the current video mode",
         parameters={
             "type": "object",
             "properties":
                 {
                     "name": {
                         "type": "string",
                         "enum": ["4K DCI", "4K UHD", "2.8K", "2K", "HD"],
                         "description": "The video mode name"
                     },
                 },
             "minProperties": 1
         },
         errors=[errors.StitcherVideoModeInvalidError, errors.StitcherVideoModeUnsupportedError]
         )
    @run_on_executor(executor='reset_executor')
    def set_video_mode(self, parameters):
        # setting the video mode is forbidden while broadcasting or recording
        if self.output_manager.has_ongoing_critical_output():
            raise errors.StitcherVideoModeChangeForbiddenError(
                "video mode cannot be changed (critical output is ongoing)")
        self.project_manager.set_resolution(parameters.get("name"))
        return self.server.reset()
