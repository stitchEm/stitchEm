import glfw
import ctypes
from threading import Event, Thread

import vs
from utils import async
from utils.settings_manager import SETTINGS
import subprocess
import time
import os
import defaults
from utils.loop import Loop

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SWITCH_DELAY = 1


class Display(object):
    """ Display automaton
    """

    def __init__(self, server_instance):
        """Init
        """
        self.DISPLAY = os.environ.get("DISPLAY")

        self.server = server_instance
        self.project_manager = self.server.project_manager

        self.pano_surfaces = None
        self.monitor = None
        self.output_window = None
        self.offscreen_window = None
        self.pano_renderer = None
        self.reset = False
        self.active = SETTINGS.display and SETTINGS.display != "none" and self.DISPLAY
        self.monitor_connected = False

        self.event_loop = Loop(0, glfw.wait_events, "X Event")
        self.event_loop.start(paused=True)

        if self.DISPLAY:
            glfw.init()
            self.monitor_connected = glfw.get_monitors()
            self.discovery_loop = Loop(2, self._discover_screen, "Screen discovery")
            self.discovery_loop.start()

    def is_active(self):
        return self.active and self.monitor_connected

    def get_surfaces(self):
        return self.pano_surfaces

    def get_renderer(self):
        return self.pano_renderer

    def reset_pano(self):
        logger.info("Opening display")

        self._pause_event_loop()

        # Open offscreen window
        self._close_offscreen_window()
        self._open_offscreen_window()
        self.width, self.height = self.project_manager.get_panorama_size()

        # Create renderer for the stitcher
        self.pano_renderer = vs.OpenGLRenderer(self._to_swig(self.offscreen_window), 2, self.width, self.height)
        self.pano_renderer.thisown = 0

        # allocate the OpenGL surfaces for the stitcher
        glfw.make_context_current(self.offscreen_window)
        self.pano_surfaces = vs.PanoSurfaceVec()
        for s in range(0, defaults.DISPLAY_BUFFER_SIZE):
            surf = vs.OpenGLAllocator_createPanoSurface(*self.project_manager.get_panorama_size())
            if not surf.ok():
                self.pano_surfaces.clear()
                self.pano_surfaces = None
                break
            ptr = vs.panoSurfaceSharedPtr(surf.release())
            self.pano_surfaces.push_back(ptr)

        glfw.make_context_current(None)

        self._unpause_event_loop()

    def close(self):
        self.close_output_window()
        self._close_offscreen_window()
        if self.DISPLAY:
            self.discovery_loop.cancel()
            glfw.terminate()

    def _unpause_event_loop(self):
        self.event_loop.unpause()

    def _pause_event_loop(self):
        self.event_loop.pause()
        glfw.post_empty_event()

    def _open_offscreen_window(self):
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.CONTEXT_ROBUSTNESS, glfw.NO_RESET_NOTIFICATION)
        glfw.window_hint(glfw.CONTEXT_RELEASE_BEHAVIOR, glfw.RELEASE_BEHAVIOR_FLUSH)
        self.offscreen_window = glfw.create_window(16, 16, "Offscreen", None, None)
        glfw.set_monitor_callback(self._monitor_event)

    def _select_monitor(self):
        self.monitor = None
        if SETTINGS.display in ["window", "windowed"]:
            return

        for monitor in glfw.get_monitors():
            logger.info("* {} : {}".format(glfw.get_monitor_name(monitor), glfw.get_video_modes(monitor)))

        for monitor in glfw.get_monitors():
            if glfw.get_monitor_name(monitor) == SETTINGS.display or SETTINGS.display == "fullscreen":
                self.monitor = monitor
                break

    def _select_video_mode(self):
        """ Selects the monitor resolution whose width is the immediately higher than the stitcher resolution
        """
        resolutions = list(reversed(glfw.get_video_modes(self.monitor)))
        self.video_mode = None
        for refresh_rate in [30, 29, 60, 59]:
            for resolution in resolutions:
                if resolution[0][0] < self.width:
                    break
                if resolution[2] == refresh_rate:
                    self.video_mode = resolution
            if self.video_mode:
                break

        if not self.video_mode:
            logger.info("No matching resolution found, using highest available")
            self.video_mode = resolutions[0]

        mode = str(self.video_mode[0][0]) + "x" + str(self.video_mode[0][1])
        rate = str(self.video_mode[2])
        subprocess.call(["xrandr", "--output", glfw.get_monitor_name(self.monitor), "--mode", mode, "--rate", rate])
        time.sleep(1)

    def _convert_rate(self, rate):
        return {29: 2997, 30: 3000, 50: 5000, 59: 5994, 60: 6000, 75: 7500}.get(rate, 3000)

    def _to_swig(self, window):
        """ Converts a ctypes-bound window to SWIG"""
        window_addr = ctypes.cast(ctypes.pointer(window),
                                  ctypes.POINTER(ctypes.c_ulong)).contents.value
        swig_object = vs.castToSwigGLFWwindow(window_addr)
        return swig_object

    def open_output_window(self):
        self._pause_event_loop()

        # Select monitor
        self._select_monitor()

        # Open window
        if self.monitor:
            # Fullscreen
            self._select_video_mode()
            x, y = glfw.get_monitor_pos(self.monitor)
            logger.info(
                "Output selected : {} on {} at {},{}".format(str(self.video_mode),
                                                             glfw.get_monitor_name(self.monitor),
                                                             x, y))
            w, h = self.video_mode[0]
            self.pano_renderer.setViewport(w, h)
            self.pano_renderer.setRefreshRate(self._convert_rate(30), self._convert_rate(self.video_mode[2]))
            if self.output_window:
                glfw.set_window_monitor(self.output_window, self.monitor, x, y, w, h, self.video_mode[2])
                glfw.show_window(self.output_window)
                self.pano_renderer.enableOutput(True)
            else:
                self._open_output_window(w, h, self.monitor, self.video_mode)
        else:
            # No monitor available or windowed
            w = self.width / 4
            h = self.height / 4
            self.pano_renderer.setViewport(w, h)

            monitor = glfw.get_primary_monitor()
            if monitor:
                rate = glfw.get_video_mode(monitor)[2]
                self.pano_renderer.setRefreshRate(self._convert_rate(30), self._convert_rate(rate))

            if self.output_window:
                self.pano_renderer.enableOutput(True)
                glfw.show_window(self.output_window)
            else:
                self._open_output_window(w, h)
                if not SETTINGS.display in ["window", "windowed"]:
                    # No monitor available
                    glfw.hide_window(self.output_window)

        self._unpause_event_loop()

    def _open_output_window(self, w, h, monitor=None, video_mode=None):
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, True)
        glfw.window_hint(glfw.CONTEXT_ROBUSTNESS, glfw.NO_RESET_NOTIFICATION)
        glfw.window_hint(glfw.CONTEXT_RELEASE_BEHAVIOR, glfw.RELEASE_BEHAVIOR_FLUSH)
        glfw.window_hint(glfw.DOUBLEBUFFER, 1)
        if video_mode:
            glfw.window_hint(glfw.REFRESH_RATE, video_mode[2])
            glfw.window_hint(glfw.RED_BITS, video_mode[1][0])
            glfw.window_hint(glfw.GREEN_BITS, video_mode[1][1])
            glfw.window_hint(glfw.BLUE_BITS, video_mode[1][2])
        self.output_window = glfw.create_window(w, h, "Output", monitor, self.offscreen_window)
        self.pano_renderer.setOutputWindow(self._to_swig(self.output_window))
        glfw.set_key_callback(self.output_window, self._key_event)
        glfw.set_window_close_callback(self.output_window, self._close_event)

    def _close_offscreen_window(self):
        if self.offscreen_window:
            glfw.make_context_current(self.offscreen_window)
            if self.pano_surfaces:
                self.pano_surfaces.clear()
                self.pano_surfaces = None
            glfw.make_context_current(None)
            self._pause_event_loop()
            glfw.destroy_window(self.offscreen_window)
            self.offscreen_window = None

    def close_output_window(self):
        if self.output_window:
            self.pano_renderer.stop()
            glfw.destroy_window(self.output_window)
            self.output_window = None

    def _key_event(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.RELEASE:
            async.defer(self.server.stop)

    def _close_event(self, window):
        async.defer(self.server.stop)

    def _monitor_event(self, monitor, event):
        name = glfw.get_monitor_name(monitor)
        if event == glfw.DISCONNECTED:
            logger.info("Unplugged monitor {}".format(name))
        else:
            logger.info("Plugged monitor {}".format(name))
        self.pano_renderer.enableOutput(False)
        if self.output_window:
            glfw.hide_window(self.output_window)
        async.delay(SWITCH_DELAY, self.open_output_window)

    def _discover_screen(self):
        if not glfw.get_monitors():
            if os.environ.get("XDG_CURRENT_DESKTOP") != "GNOME":
                # If no monitors detected, updates list with xrandr
                cmd = "xrandr --query | grep ' connected'"
                result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT).communicate()
                if result[0] != "":
                    subprocess.call(["xrandr", "--auto"])
                    if not self.monitor_connected:
                        # If we boot without a monitor connected, and then plug one, the first display switch fails for an unknown reason.
                        # The current solution is to start with the pure cuda stitching loop, and then switch to opengl
                        # Subsequent plug/unplug operation are fine, though
                        self.monitor_connected = True
                        async.defer(self.server.reset)
