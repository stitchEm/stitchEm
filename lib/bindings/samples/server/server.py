import gc
import logging
import os
import pprint
import signal
import blinker
import socket
import subprocess
import sys
import datetime

from API import version
from utils.settings_manager import SETTINGS
import tornado
import tornado.options
import tornado.websocket
from tornado import httpserver
from tornado.ioloop import IOLoop
from tornado.web import Application, url
from transitions.extensions import LockedMachine as Machine
import defaults
import utils.filesystem


# Init

def set_env():
    """
    Set the system path (for modules) and LD_LIBRARY_PATH (for dynamic
    libraries)
    Note :  The only way to define LD_LIBRARY_PATH is to launch a new instance
            of the server. We're only using that when debugging locally
    """
    sys.path.insert(0, SETTINGS.lib_path)
    if "LD_LIBRARY_PATH" not in os.environ:
        subprocess_env = os.environ.copy()
        subprocess_env["LD_LIBRARY_PATH"] = SETTINGS.lib_path
        try:
            subprocess.check_call(
                ["/usr/bin/python"] + sys.argv, stderr=subprocess.STDOUT, env=subprocess_env)
        except KeyboardInterrupt:
            pass
        return False
    return True


def create_directories():
    """Make sure the default directories exists.
    """
    utils.filesystem.create_dir(SETTINGS.log_path)
    utils.filesystem.create_dir(defaults.USER_RECORDINGS_PATH)

    utils.filesystem.copy_tree(
        defaults.SYSTEM_PRESETS_DIR_PATH,
        defaults.USER_PRESETS_DIR_PATH,
        force_copy=SETTINGS.force_default,
        only_structure=True,
        destroy_copies=True)


version.set_globals()
if not set_env():
    sys.exit(-1)
create_directories()

# Set up logging

logging.getLogger().setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
try:
    from utils import log
except ImportError:
    sys.stderr.write("ERROR: cannot import dependencies. Aborting\n")
    raise

start_string = "##################################################################################################\n"
start_string = start_string + version.BUILD_VERSION + defaults.SESSION_UUID

log.set_python_log(SETTINGS.loglevel, SETTINGS.python_log_file, SETTINGS.logrotate)

logger.info(start_string)
logger.info("\n\n")
logger.info("Python version: " + sys.version)
logger.info("Tornado version: " + tornado.version)
logger.info("Lib path: " + SETTINGS.lib_path)
logger.info("Server path: " + defaults.FILE_PATH)
logger.info("Logs path: " + SETTINGS.log_path)
logger.info(pprint.pformat(vars(os.environ)) + "\n\n")
logger.info(pprint.pformat(vars(SETTINGS)) + "\n\n")

#

from system import drive_manager
from API import audio_api, box_api, camera_api, stitcher_api, algorithm_api, social_api, handlers, refs
import utils.update_checker
from utils.flash_checker import FLASH_CHECKER
from clientmessenger import CLIENT_MESSENGER
from system import network_monitor

try:
    from debug.debug_api import DebugAPI
    from debug.debugging import Debugging
except ImportError:
    logger.debug("Disabling debug handler")
    DebugAPI = None
    Debugging = None

# default values
EXIT_MESSAGE = ""
EXIT_CODE = 0


def make_application_API(server, video_stitcher, output_manager, preset_manager, project_manager, camera, verbose):
    """ Builds the tornado application server
    """
    settings = dict(
        debug=True,
        compress_response=True
    )

    # prevent tornado from logging
    settings["log_function"] = lambda r: None

    api_handlers = {
        "camera": camera_api.CameraAPI,
        "stitcher": stitcher_api.StitcherAPI,
        "box": box_api.BoxAPI,
        "algorithm": algorithm_api.AlgorithmAPI,
        "audio": audio_api.AudioAPI,
        "social": social_api.SocialAPI
    }

    if DebugAPI is not None:
        api_handlers["debug"] = DebugAPI

    return Application([
        url(r"/", handlers.IndexPageHandler),
        url(r"/firmware/(.*)", handlers.StaticFileHandler, {"path": defaults.DIR_PATH + "/firmware"}),
        url(r"/log/(.*)", handlers.StaticTextFileHandler, {"path": SETTINGS.log_path}),
        url(r"/commands/execute", handlers.CommandHandler, {
            "apiHandlers": api_handlers,
            "extra": {
                "server": server,
                "video_stitcher": video_stitcher,
                "camera": camera,
                "output_manager": output_manager,
                "preset_manager": preset_manager,
                "project_manager": project_manager,
                "verbose": verbose
            }
        })],
        **settings
    )


def save_pid():
    pid = str(os.getpid())
    file(defaults.PID_PATH, "w").write(pid)

def save_version():
    file(defaults.VERSION_LOG_PATH, "w").write(version.BUILD_VERSION)

class Server(object):
    """ Main automaton

        - Instantiates and holds camera, stitcher and output manager
        - Starts Tornado main loop

    """

    def __init__(self):

        # State Machine

        states = ["Initial",
                  "Starting",  # Creating all the objects
                  "Running",  # Tornado loop is launched and active
                  "Stopped",
                  "Failed"]

        transitions = [

            {"source": "Initial",
             "trigger": "t_start",
             "before": "_start",
             "dest": "Starting"},

            {"source": "Starting",
             "trigger": "t_run",
             "dest": "Running"},

            {"source": "Starting",
             "trigger": "t_stop",
             "before": "_stop",
             "dest": "Stopped"},

            {"source": "Running",
             "trigger": "t_stop",
             "before": "_stop",
             "dest": "Stopped"},
        ]

        self.machine = Machine(
            name="Server", model=self, states=states, transitions=transitions,
            initial="Initial", async=True)

    def _start(self):
        self._start_application()
        blinker.signal("box_flash_detected").connect(self._box_flash_detected)
        FLASH_CHECKER.check_flash_file()
        blinker.signal("update_available").connect(self._update_available)
        self.update_checker = utils.update_checker.UpdateChecker(SETTINGS.update_info_url)
        self.network_monitor_output = network_monitor.NetworkMonitor(defaults.NETWORK_OUTPUT_INTERFACE, True,
                                                                     "streaming")

    def _stop(self):
        self.update_checker.stop()
        self.network_monitor_output.stop()
        self.camera.t_terminate()
        self.stitcher.terminate()
        self.project_manager.terminate()
        self.output_manager.terminate()
        drive_manager.DRIVE_MANAGER.terminate()
        self.display.close()
        gc.collect()

    def _start_application(self):
        """ Creates and initializes objects"""

        log.set_vs_log(SETTINGS.loglevel, SETTINGS.logrotate, start_string)

        save_pid()
        save_version()

        try:
            import stitcher
            import camera
            import display
            from output.output_manager import OutputManager
            from project_manager import ProjectManager
            from preset_manager import PresetManager
        except ImportError:
            logger.error("cannot import dependencies. Aborting\n")
            raise

        self.preset_manager = PresetManager()
        self.camera = camera.Camera(self)
        if SETTINGS.ptv is not None:
            self.project_manager = ProjectManager(SETTINGS.ptv)
        else:
            self.project_manager = ProjectManager(defaults.USER_DEFAULT_PTV_PATH)

        self.display = display.Display(self)
        self.stitcher = stitcher.Stitcher(self, stitcher_api.StitcherAPI.stitcher_executor, self.display)
        self.output_manager = OutputManager(self.stitcher)

        app = make_application_API(self, self.stitcher, self.output_manager, self.preset_manager, self.project_manager,
                                   self.camera, SETTINGS.verbose)
        self.http_server = httpserver.HTTPServer(app)

        self.camera.t_init()

        if Debugging is not None:
            self.debugging = Debugging(self.output_manager, self.preset_manager)
            blinker.signal("start_debug_failed").connect(self.stitcher.terminate)

        CLIENT_MESSENGER.send_event("server_started",
                                    {"date": str(datetime.datetime.now()), "session": defaults.SESSION_UUID})

        try:
            refs.load()
            self.http_server.listen(SETTINGS.port)
            self.t_run()
        except socket.error, msg:
            global EXIT_MESSAGE
            global EXIT_CODE
            EXIT_MESSAGE = "Socket error: " + str(msg)
            EXIT_CODE = -1
            self.stop()

    def sigint(self, signum=None, frame=None):
        global EXIT_MESSAGE
        EXIT_MESSAGE = "SIGINT received..."
        self.stop()

    def sighup(self, signum=None, frame=None):
        """ Restarts libvideostitch log
        """
        logging.info("SIGHUP received, restarting libvideostitch log")
        loglevel = SETTINGS.loglevel
        utils.log.set_vs_log(loglevel, True)

    def force_stop(self, signum=None, frame=None):
        global EXIT_MESSAGE
        EXIT_MESSAGE = "Stopped..."
        self.stop()

    def stop(self):
        """ Stops the server

        Notes:
            As this function terminates the stitcher, it should not be called from the executor running
            the stitching process to prevent deadlocks.

        """
        self.t_stop()
        self.http_server.stop()
        IOLoop.current().stop()

    def reset(self, force_default=False):
        """ Resets the server with the given parameters

        Notes:
            As this function terminates the stitcher, it should not be called from the executor running
            the stitching process to prevent deadlocks.

        """
        if self.camera.is_Connected():
            self.camera.stop_streams()
        self.stitcher.terminate()
        self.project_manager.reset(force_default)
        self.output_manager.reset()
        # We (unfortunately) rely on garbage collection to make sure that c++ objects are deleted
        # So take care not to keep hidden references of objects that are supposed to be destroyed when we reset, like in the API objects for instance
        # In general, objects created by Server and passed as 'extra' are transient and can be kept, but none of their properties should be.
        gc.collect()
        if self.camera.is_Connected():
            self.stitcher.t_camera_connected()
            self.camera.start_streams()

    def _update_available(self, sender):
        CLIENT_MESSENGER.send_event("update_available", self.update_checker.update_info)

    def _box_flash_detected(self, sender):
        CLIENT_MESSENGER.send_event("box_flash_detected")


# Global server instance
SERVER = Server()


def start():
    SERVER.t_start()

    signal.signal(signal.SIGINT, SERVER.sigint)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, SERVER.sighup)
    IOLoop.current().start()
    logger.info("**********************************************************************")
    logger.info(EXIT_MESSAGE)
    logger.info("**********************************************************************")
    sys.exit(EXIT_CODE)
