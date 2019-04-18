import vs

import errors
import gc
from output import WriterOutput

class PaOutput(WriterOutput):

    def __init__(self, stitcher, name, critical=False, preserved=False):
        super(PaOutput, self).__init__(stitcher, name, critical, preserved)
        self.writer_name = name
        # If you need to test the audio HDMI
        #self.pa_devices = ["HDA NVidia: HDMI 0 (hw:2,3)", "HDA NVidia: HDMI 1 (hw:2,7)", "HDA NVidia: HDMI 2 (hw:2,8)", "HDA NVidia: HDMI 3 (hw:2,9)"]
        # Line out
        self.pa_devices = ["HDA Intel PCH"]

    #override additional preset to avoid overwriting channel layout
    def setAdditionalPreset(self, additional_preset):
        if ("channel_layout" in additional_preset):
            del additional_preset["channel_layout"]
        super(PaOutput, self).setAdditionalPreset(additional_preset)


    def _start(self, preset=None, preserve=False):
        self._load_preset(preset)
        self._add_writer()


    def _stop(self):
        for pa_device in self.pa_devices:
            self._remove_writer(pa_device)
        self.t_writer_completed()

    def _remove_writer(self, writer_name):
        """Stops the stream
        """
        if self.has_audio:
            success = self.stitcher.project_manager.controller.removeAudioOutput(writer_name)
            if not success:
                raise errors.InternalError("Cannot remove audio writer {}".format(writer_name))

        self.shared_writer = None
        self.shared_video = None
        self.timer.reset()
        gc.collect()
        self.flush_writer_events()

    def _add_writer(self):
        for pa_device in self.pa_devices:
            self.ptv["name"] = pa_device
            self.config = self.ptv.to_config()
            writer = self._create_writer(self.stitcher.project_manager.panorama, self.stitcher.project_manager.controller, self.config, pa_device)

            self.shared_writer = vs.writerSharedPtr(writer.release())
            self.connect_writer_events(self.shared_writer)
            self.shared_video = vs.videoWriterSharedPtr(self.shared_writer)
            shared_audio = vs.audioWriterSharedPtr(self.shared_writer)
            self.has_audio = True
            if self.shared_video is not None and not self.stitcher.stitch_output.addWriter(self.shared_video):
                raise errors.InternalError("Cannot add video writer to stitcher")
            if shared_audio is not None and not self.stitcher.project_manager.controller.addAudioOutput(shared_audio):
                raise errors.InternalError("Cannot add audio writer to stitcher")

            self.timer.start()
            gc.collect()

    def get_latency(self):
        return self._get_latency()
