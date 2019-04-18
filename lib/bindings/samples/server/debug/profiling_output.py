import threading

import errors
import vs
import logging
import gc

from blinker import signal
from utils import performance
from output.output import Output


# We need to be able to load (not run) vs_server on windows to generate the documentation.
# So we're skipping non-windows imports
try:
    import psutil
except ImportError:
    pass

PROFILING_STITCH_FORMAT = vs.NV12


class ProfilingOutput(Output):
    """Profiling output
    """

    def __init__(self, stitcher, name="profiling", critical=False, preserved=False):
        super(ProfilingOutput, self).__init__(stitcher, name, critical, preserved)

        self.writer = None
        self.pid = psutil.Process()

    def reset(self):
        self._transition_check()
        self.pid.cpu_percent(interval=None)
        vs.Output_reset(self.writer.object())

    def _start(self, profiling_time=0, preserve=False):
        # Todo I don't like that it's created differently from other outputs here, but for now I left it like this
        panorama = self.stitcher.project_manager.panorama
        self.writer = vs.Output_profiling(self.name,
                                          panorama.width,
                                          panorama.height,
                                          self.stitcher.project_manager.controller.getFrameRateFromInputController(),
                                          PROFILING_STITCH_FORMAT)
        if self.writer is None:
            raise errors.InternalError()

        self.shared_writer = vs.writerSharedPtr(self.writer.object())
        self.shared_video  = vs.videoWriterSharedPtr(self.shared_writer)
        self.has_audio = False
        if self.shared_video is not None and not self.stitcher.stitch_output.addWriter(self.shared_video):
            raise errors.InternalError("Cannot add profiling writer to stitcher")
        if profiling_time > 0:
            threading.Timer(profiling_time, self.t_stop).start()
        self.pid.cpu_percent(interval=None)
        #jump automatically from starting state to started state
        self.t_writer_ok()

    def _stop(self):
        self.fps = vs.Output_getFps(self.writer.release())
        self.writer = None
        logging.info("fps is %f:" % self.fps)
        logging.info("cpu_util is %d" % self.pid.cpu_percent(interval=None))
        cuda = performance.getCudaInfo()
        logging.info("gpu_util is %d" % int(cuda['utilization.gpu']))
        logging.info("enc_util is %s" % cuda['utilization.enc'])
        success = self.stitcher.stitch_output.removeWriterNoGIL(self.name)
        signal("profiling_stopping").send()
        if not success:
            raise errors.InternalError("Cannot remove writer")
        self.shared_video = None
        self.shared_writer = None
        gc.collect()
        #jump automatically from stopping state to stopped state
        self.t_writer_completed()


    def get_statistics(self):
        cuda = performance.getCudaInfo()
        self._transition_check()
        if self.writer is not None:
           self.fps = vs.Output_getFps(self.writer.object())
        return {"fps": self.fps,
                "cpu": self.pid.cpu_percent(interval=None),
                "gpu": float(cuda['utilization.gpu']),
                "enc": float(cuda['utilization.enc'])}
