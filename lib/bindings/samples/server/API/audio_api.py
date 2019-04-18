from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor
from API.handlers import APIHandler
from API.schema import api
import errors

class AudioAPI(APIHandler):
    """REST interface to the Box
    """
    executor = ThreadPoolExecutor(1)

    def __init__(self, extra):
        """Init
        """
        super(AudioAPI, self).__init__(extra)
        self.server = extra["server"]
        self.project_manager = extra["project_manager"]
        self.output_manager = extra["output_manager"]

    @api(name="ListSources",
         endpoint="audio.list_sources",
         description="Lists audio sources",
         result={
             "type": "object",
             "properties":
                 {
                     "audio_sources": {
                         "type": "array",
                         "items": {
                             "type": "object",
                             "properties": {
                                 "name": {
                                     "$ref": "AudioSourceName"
                                 },
                                 "support_delay": {
                                     "type": "boolean"
                                 },
                                 "layouts": {
                                     "type": "array",
                                     "items": {
                                        "$ref": "AudioSourceLayout"
                                    }
                                 }
                             }
                         }
                     }
                 },
             "minProperties": 1
         }
         )
    @run_on_executor
    def list_sources(self, parameters):
        sources = []
        project_audio_sources = self.project_manager.list_audio_sources()
        for source_name in project_audio_sources:
            sources.append({
                "name": source_name,
                "support_delay": project_audio_sources[source_name]["input"]
                                 in self.project_manager.AUDIO_INPUTS_SUPPORTING_DELAY,
                "layouts": project_audio_sources[source_name]["layouts"]})
        return {"audio_sources": sources}

    @api(name="GetSource",
         endpoint="audio.get_source",
         description="Gets current audio source",
         result={
             "type": "object",
             "properties":
                 {
                     "source": {
                         "$ref": "AudioSourceName"
                     },
                     "layout": {
                         "$ref": "AudioSourceLayout"
                     }
                 },
         }
         )
    @run_on_executor
    def get_source(self, parameters):
        return {"source": self.project_manager.get_audio_source(), "layout": self.project_manager.get_audio_layout()}

    @api(name="SetSource",
         endpoint="audio.set_source",
         description="Sets the current audio source",
         parameters={
             "type": "object",
             "properties":
                 {
                     "source": {
                         "$ref": "AudioSourceName"
                     },
                     "layout": {
                         "$ref": "AudioSourceLayout"
                     }
                 },
             "required": ["source", "layout"]
         }
         )
    @run_on_executor
    def set_source(self, parameters):
        # setting the video mode is forbidden while broadcasting or recording
        if self.output_manager.has_ongoing_critical_output():
            raise errors.AudioSourceChangeForbiddenError(
                "audio source cannot be changed (critical output is ongoing)")
        self.project_manager.set_audio_source(str(parameters.get("source")), str(parameters.get("layout")))
        self.server.reset()

    @api(name="GetAudioDelay",
         endpoint="audio.get_delay",
         description="Gets the current delay for the specified source",
         parameters={
             "type": "object",
             "properties":
                 {
                     "source": {
                         "$ref": "AudioSourceName"
                     }
                 },
             "minProperties": 1
         },
         result={
             "type": "object",
             "properties":
                 {
                     "delay": {
                         "type": "number",
                         "description": "The audio delay in seconds"
                     }
                 }
         }
         )
    @run_on_executor
    def get_delay(self, parameters):
        return {"delay": self.project_manager.get_audio_delay(str(parameters.get("source")))}

    @api(name="SetAudioDelay",
         endpoint="audio.set_delay",
         description="Sets the current audio delay for the specified source",
         parameters={
             "type": "object",
             "properties":
                 {
                     "source": {
                         "$ref": "AudioSourceName"
                     },
                     "delay": {
                         "type": "number",
                         "minimum": -1.0,
                         "maximum": 1.0,
                         "description": "The audio delay in seconds"
                     }
                 }
         }
         )
    @run_on_executor
    def set_delay(self, parameters):
        self.project_manager.set_audio_delay(str(parameters.get("source")), parameters.get("delay"))


