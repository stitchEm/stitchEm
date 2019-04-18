import logging
import threading
import time

from os import path as osp

from transitions.extensions import LockedMachine as Machine
from transitions import State as State
from blinker import signal

import defaults
import errors
import vs
import utils
import glfw

from algorithm.algorithm_manager import AlgorithmManager
from clientmessenger import CLIENT_MESSENGER
from utils.settings_manager import SETTINGS
from utils import async

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

FAIL_SLEEP = 0.25


class Stitcher(object):
    """ Stitcher automaton

        - Responsible for the stitching process
        - Holds references to streams (preview, broadcast)
        - Holds references to recorders (input, output)
    """

    def __init__(self, server_instance, stitcher_executor, display_):
        """Init
        """
        self.server = server_instance
        self.stitcher_executor = stitcher_executor

        self.algorithm_manager = None
        self.has_audio = None
        self.project_manager = self.server.project_manager
        self.display = display_
        self.old_submit = self.stitcher_executor.submit

        self.overlay_window = None
        self.overlay_config = None
        self.overlay = None

        # State Machine

        states = [
            # The stitcher is waiting for the camera to connect
            State(name="WaitingForCamera", ignore_invalid_triggers=True),
            # Stitcher running, no errors
            State(name="Running", ignore_invalid_triggers=True),
            # Stitcher running, but producing errors (EOF, etc..)
            State(name="Failing", ignore_invalid_triggers=True),
        ]

        transitions = [

            {"source": "WaitingForCamera",
             "trigger": "t_camera_connected",
             "dest": "Failing"},

            {"source": "Running",
             "trigger": "t_fail",
             "dest": "Failing"},

            {"source": "Failing",
             "trigger": "t_resume",
             "dest": "Running"},

            {"source": ["Running", "Failing"],
             "trigger": "t_camera_connected",
             "conditions": "_ignore",
             "dest": "Failing"},

            {"source": ["Running", "Failing"],
             "trigger": "t_camera_disconnected",
             "before": "terminate",
             "dest": "WaitingForCamera"},

            {"source": ["Running", "Failing"],
             "trigger": "t_terminate",
             "dest": "WaitingForCamera"},
        ]

        self.machine = Machine(
            name="Stitcher", model=self, states=states,
            transitions=transitions, initial="WaitingForCamera", async=True)

        # Signals

        self._connect_signals()
        self._load_plugins()
        self.run_loop = None
        self.native_loop = None

    def _connect_signals(self):
        signal("camera_connected").connect(self.t_camera_connected)
        signal("camera_disconnected").connect(self.t_camera_disconnected)
        signal("camera_video_fail").connect(self.t_camera_disconnected)
        signal("algorithm_running_success").connect(self._reset_executor)

    def _submit(self, fn, *args, **kwargs):
        self.native_loop.Lock()
        result = self.old_submit(fn, *args, **kwargs)
        self.native_loop.Unlock()
        return result

    def _reset_executor(self, sender, panorama, output):
        self.stitcher_executor.submit(self.reset_panorama, sender, panorama, output)

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------

    def _ignore(self, sender=None):
        return False

    def on_enter_WaitingForCamera(self, sender=None):
        self._cleanup()

    def on_exit_WaitingForCamera(self, sender=None):
        """ Create stitcher"""
        self._create()

    def start_streams(self):
        signal("stitching_running").send()

    def on_enter_Running(self, sender=None):
        self.stitcher_executor.submit(self.start_streams).result()
        if SETTINGS.enable_ev_compensation:
            # Note, if stitcher fails and algorithm won't be completed - it won't be rescheduled
            # So we want to start it again when stitcher is running
            self.algorithm_manager.start_exposure_compensation()

    def on_exit_Running(self, sender=None):
        signal("stitching_failing").send(preserve=SETTINGS.output_recovery_enabled)

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------

    def _load_plugins(self):
        if vs.loadPlugins(osp.join(SETTINGS.lib_path, "plugins")) == 0:
            raise errors.StitcherError("Parsing: could not load any plugin")

    def _create(self):
        """ Creates the controller and the stitcher
        """

        status = vs.checkDefaultBackendDeviceInitialization()
        if not status.ok():
            logger.error("CUDA device Initialization failed. Reason {}".format(status.getErrorMessage()))
            raise errors.StitcherError("CUDA device Initialization failed")

        try:
            self.project_manager.create_controller()
        except (errors.StitcherError, errors.AudioError, errors.ParserError) as error:
            logger.warn("current configuration seems corrupted : {} "
                        "Switching to default configuration.".format(error))
            CLIENT_MESSENGER.send_event(defaults.RESET_TO_DEFAULT_MESSAGE)
            utils.async.defer(self.server.reset, True)
            return

        status = self.project_manager.controller.createStitcher()
        if not status.ok():
            raise errors.StitcherError("Cannot create stitcher: " + str(status.getErrorMessage()))

        self._start_stitching_thread()

        self.algorithm_manager = AlgorithmManager(self.project_manager.controller)

        self.renderer = None
        if self.display.is_active():
            self.display.reset_pano()
            self.display.open_output_window()
            surfaces = self.display.get_surfaces()
            if surfaces is None:
                raise errors.StitcherError("Error creating destination surface")
            self.renderer = self.display.get_renderer()
            self.renderer.thisown = 0
            self.stitcher_executor.submit = self._submit

            #overlay create
            self.overlay_config = self.project_manager.project_config.has("overlay")
            if self.overlay_config :
                glfw.default_window_hints()
                glfw.window_hint(glfw.VISIBLE, False)
                glfw.window_hint(glfw.CONTEXT_ROBUSTNESS, glfw.NO_RESET_NOTIFICATION)
                glfw.window_hint(glfw.CONTEXT_RELEASE_BEHAVIOR, glfw.RELEASE_BEHAVIOR_FLUSH)
                self.overlay_window = glfw.create_window(16, 16, "", None, self.display.offscreen_window)
                glfw.make_context_current(self.overlay_window)
                self.overlay = vs.Compositor(self.display._to_swig(self.overlay_window),  self.project_manager.panorama, self.project_manager.controller.getFrameRateFromInputController())
                glfw.make_context_current(None)
        else:
            surfaces = vs.PanoSurfaceVec()
            surf = vs.OffscreenAllocator_createPanoSurface(self.project_manager.panorama.width,
                                                           self.project_manager.panorama.height,
                                                           "StitchOutput")
            if not surf.ok():
                raise errors.StitcherError("Error creating destination surface " + \
                                           str(surf.status().getErrorMessage()))
            dontmemleak = vs.panoSurfaceSharedPtr(surf.release())
            # SWIG create a proxy object with an empty deleter
            # when passing directly the pointer to the vector object :(
            # DON'T TRY TO FACTORIZE THE PREVIOUS LINE OR MEMLEAK
            surfaces.push_back(dontmemleak)
            self.stitcher_executor.submit = self.old_submit

        self.stitch_output = self.project_manager.controller.createAsyncStitchOutput(
            surfaces, vs.PanoRendererPtrVector(), vs.OutputWriterPtrVector())

        if self.display.is_active():
            self.stitch_output.addRenderer(vs.panoRendererSharedPtr(self.renderer))
            self.native_loop = vs.StitchLoop(self.display._to_swig(self.display.offscreen_window), self.project_manager.controller, self.stitch_output.object())

        if self.overlay:
            dontmemleak = vs.overlayerSharedPtr(self.overlay)
            # SWIG create a proxy object with an empty deleter
            # when passing directly the pointer to the setCompositor function :(
            # DON'T TRY TO FACTORIZE THE PREVIOUS LINE OR MEMLEAK
            self.stitch_output.setCompositor(dontmemleak)
        self.run_loop.set()

        # To be deprecated
        self.has_audio = False

    def _start_stitching_thread(self):
        self.run_loop = threading.Event()
        self.thread = threading.Thread(target=self._stitch_loop, name="Stitcher")
        self.thread.start()

    # Stitching process management

    def _cleanup(self):
        """Makes sure that all the lib objects are released to prevent assert at exit time
        """
        if self.display.is_active():
            self.display.close_output_window()
            self.stitch_output.removeRenderer("OpenGLRenderer")
        self.stitch_output = None
        self.algorithm_manager = None

    def _check_status(self, status):
        current_state = self.state
        if status is not None:
            if current_state == "WaitingForCamera":
                time.sleep(FAIL_SLEEP)
                return None
            if status.ok():
                if current_state == "Failing":
                    async.defer(self.t_resume)
                    return status
            else:
                msg = status.getErrorMessage()
                if current_state == "Running":
                    logger.error("Stitching failed : " + msg)
                    async.defer(self.t_fail)
                    return status
                # unrecoverable GPU failure. Server need to be resetted
                elif self.state == "Failing" and status.getOriginString() == "GPU":
                    logger.error("Stitching critical failed : " + msg)
                    CLIENT_MESSENGER.send_error(errors.StitcherError("Internal GPU error: " +
                                                                     status.getErrorMessage()))
                    utils.async.defer(self.server.reset)
                    return status

        # No state change
        return None

    def _stitch_loop(self):
        """Stitcher runner job (blocking while stitcher is starved)
        """
        logger.info("starting loop...")
        self.run_loop.wait()

        if self.display.is_active():
            """ When displaying on HDMI, the stitch loop runs in C++.
                This loop checks the status of the native loop on a regular basis
            """
            self.native_loop.Start()
            old_status = None
            while self.run_loop.is_set():
                algodata = self.algorithm_manager.get_next_algorithm(self.project_manager.panorama)
                self.native_loop.setAlgorithm(algodata[0], algodata[1])
                status = self.native_loop.StitchStatus()
                if old_status is None or old_status.ok() != status.ok():
                    old_status = self._check_status(status)
                time.sleep(1 / 1000.)
            self.native_loop.Stop()
        else:
            """ When not ouptuting to HDMI, the stitcher_executor fires the stitch command
            """
            while self.run_loop.is_set():
                status = self.stitcher_executor.submit(self._stitch).result()
                self._check_status(status)
                time.sleep(1 / 1000.)

        logger.info("leaving loop...")

    def _stitch(self):
        """Stitch the input frames into the output. No python thread blocking.
        """
        algodata = self.algorithm_manager.get_next_algorithm(self.project_manager.panorama)
        status = None
        try:
            status = vs.stitchAndExtractNoGIL(
                self.project_manager.controller,
                self.stitch_output.object(),
                algodata[1],
                algodata[0],
                True)
        except Exception as e:
            logger.warning(e)
        return status

    def reset_panorama(self, sender, updater, output):
        # Output is passed so that we keep a reference to the underlying object. Please do not remove this unused parameter.
        # Get updater function if it's updater, otherwise - pass panorama
        updater_param = updater.getCloneUpdater() if isinstance(updater, vs.PanoramaDefinitionUpdater) else updater
        status = self.project_manager.controller.updatePanorama(updater_param)
        if not status.ok():
            CLIENT_MESSENGER.send_error(errors.StitcherError("Cannot update the panorama. Reason: " +
                                                             status.getErrorMessage()))
            return

        self.project_manager.update("pano", self.project_manager.get_save_panorama().serialize())

    def terminate(self, sender=None):
        if self.run_loop and self.run_loop.is_set():
            self.run_loop.clear()
            logger.info("Joining run loop...")
            self.thread.join()
            logger.info("Done")
        if self.algorithm_manager is not None:
            self.algorithm_manager.cancel_running_algorithms()
        self.t_terminate()
