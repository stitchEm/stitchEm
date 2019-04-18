import logging
from concurrent.futures import ThreadPoolExecutor
from os import listdir
from os import path as osp
import netifaces

import defaults
import errors
import utils.async
from system.drive_manager import DRIVE_MANAGER
from API.handlers import APIHandler
from tornado.concurrent import run_on_executor
from transitions import MachineError
from API import stitcher_api
from clientmessenger import CLIENT_MESSENGER
from utils.settings_manager import SETTINGS
from debug.systemmonitor import MONITOR


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

LOG_FOLDER_URL = "log"
LOG_PYTHON_LOG_NAME = "python"


class DebugAPI(APIHandler):
    """REST interface for debugging / testing
    """
    log_executor = ThreadPoolExecutor(1)
    stitcher_executor = stitcher_api.StitcherAPI.stitcher_executor
    executor = ThreadPoolExecutor(1)

    def __init__(self, extra):
        """Init
        """
        self.server = extra["server"]
        self.stitcher = extra["video_stitcher"]
        self.project_manager = extra["project_manager"]
        self.output_manager = extra["output_manager"]
        self.preset_manager = extra["preset_manager"]
        self.camera = extra["camera"]
        self.monitor = MONITOR

    @run_on_executor
    def list_logs(self, parameters=None):
        from utils.log import LOG_FILES
        logs = {}
        for logName, logInfo in LOG_FILES.iteritems():
            logs[logName] = "/{}/{}".format(LOG_FOLDER_URL, logInfo[0])
        if not SETTINGS.python_log_file:
            del logs[LOG_PYTHON_LOG_NAME]
        return logs

    @run_on_executor
    def stop_server(self, parameters=None):
        utils.async.delay(0.5, self.server.force_stop)

    @run_on_executor(executor='log_executor')
    def log(self, parameters=None):
        logger.info(parameters.get("message"))

    @run_on_executor
    def send_error(self, parameters=None):
        CLIENT_MESSENGER.send_error(errors.VSError(parameters["message"]))

    @run_on_executor
    def send_event(self, parameters=None):
        CLIENT_MESSENGER.send_event(parameters["name"], parameters.get("payload"))

    @run_on_executor
    def clear_messages(self, parameters=None):
        CLIENT_MESSENGER.clear()


    @run_on_executor
    def get_drive_list(self, parameters):
        """Returns the available storage devices

        Returns:
            - ``InternalError``
            - Otherwise: a list with 0 or more drives.

        Note:
            Calling this function can create some overhead.
        """
        return self.monitor.storage()

    @run_on_executor
    def get_drive_info(self, parameters):
        """Returns the storage drive information

        Returns:
            - ``InternalError``
            - Structure::

                {
                    'results': {
                        'used': string,
                        'percent': float,
                        'free': string,
                        'label': string,
                        'fs': string,
                        'device': string,
                        'mountpoint': string,
                        'total': string
                    }
                }

        Note:
            Calling this function can create some overhead
        """
        try:
            drive = parameters['drive']
        except:
            raise errors.InvalidParameter('No specified drive')
        mp_stats = self.monitor.drive_info(drive)
        if mp_stats is None:
            raise errors.InternalError(
                'No information found for drive {}'.format(parameters))

    @run_on_executor
    def get_available_percentage(self, parameters):
        """Returns the available size of a selected drive in percentage

        Args:
            parameters(JSON): Should containt the field 'mountpoint'
            specifying the target drive

        Returns:
            - ``InternalError``
            - Otherwise: the available percentage of the drive.
        Note:
            Calling this function can create some overhead
        """
        return self.monitor.get_available_percentage(parameters)

    def get_network_adapters(self, parameters):
        """Returns the available network adapters.
        Returns:
            - ``InternalError``
            - Otherwise: a list of 0 or more network adapters.
        """
        return netifaces.interfaces()

    @run_on_executor
    def get_hardware_status(self, parameters=None):
        config = {"hardware": self.monitor.status()}
        return config


    # Camera

    @run_on_executor
    def connect_camera(self, parameters=None):
        self.server.camera.t_force_connect()

    @run_on_executor
    def disconnect_camera(self, parameters=None):
        try:
            self.server.camera.t_disconnect()
        except MachineError:
            pass

    @run_on_executor
    def simulate_camera_calibration_files(self, parameters=None):
        with open("./test/test_data/calibration_rig_parameters.json", "r") as rig_params_file:
            rig_parameters = rig_params_file.read()
        self.camera.rig_parameters = rig_parameters


    @run_on_executor
    def force_update_firmware(self, parameters):
        return self.camera.force_update_firmware(str(parameters.get("name")) if parameters is not None else None)

    @run_on_executor
    def get_firmware_list(self, parameters=None):
        files = [wfile for wfile in listdir(defaults.FIRMWARE_DIR_PATH)
                 if osp.splitext(wfile)[1] == defaults.FIRMWARE_EXTENSION]
        return {"entries": files}

    # Stitcher and profiler

    @run_on_executor(executor='stitcher_executor')
    def start_profiling(self, parameters=None):
        """Starts the profiling.
        """
        self.output_manager.get_output("profiling").start()

    @run_on_executor(executor='stitcher_executor')
    def stop_profiling(self, parameters=None):
        """Stop the profiling.
       """
        self.output_manager.get_output("profiling").stop()

    @run_on_executor(executor='stitcher_executor')
    def reset_profiling(self, parameters=None):
        """Reset the profiling.
        """
        self.output_manager.get_output("profiling").reset()

    @run_on_executor(executor='stitcher_executor')
    def get_status(self, parameters=None):
        """Get the debugging result.

        result={
            "inputs":    {"latency": "RTMP latency"},
            "streaming": {"latency": "RTMP latency"},
            "preview":   {"latency": "RTMP latency"},
            "profiling": {
                "fps": "frame rate",
                "cpu": "CPU usage",
                "gpu": "GPU usage",
                "enc": "NVENC usage"
            }
        }
        """
        
        input_status     = { "latency" : self.project_manager.get_latency() }
        streaming_status = { "latency" : self.output_manager.get_output("stream").get_latency() }
        preview_status   = { "latency" : self.output_manager.get_output("preview").get_latency() }
        profiling_status = self.output_manager.get_output("profiling").get_statistics()
        status = {"inputs":    input_status,
                  "streaming": streaming_status,
                  "preview":   preview_status,
                  "profiling": profiling_status}
        return status


    @run_on_executor(executor='stitcher_executor')
    def start_stream_with_json_preset(self, parameters):
        self.output_manager.start_output("stream", parameters)

    @run_on_executor(executor='stitcher_executor')
    def add_managed_drive(self, parameters):
        drive_path = parameters.get('drive_path')
        if not drive_path:
            return
        DRIVE_MANAGER.add_managed_drive(drive_path)

    # Settings
    def get_setting(self, key):
        return getattr(SETTINGS, key)

    def set_setting(self, parameter):
        for key, value in parameter.iteritems():
            setattr(SETTINGS, key, value)

    # exposure compensation
    """
    @api(name="StartExposureCompensation",
         endpoint="algorithm.start_exposure_compensation",
         description="Start exposure compensation algorithm",
         errors=[errors.AlgorithmError, errors.StitcherError]
         )
    """
    @run_on_executor
    def start_exposure_compensation(self, parameters=None):
        if self.stitcher.algorithm_manager:
            self.stitcher.algorithm_manager.start_exposure_compensation()

    """
    @api(name="StopExposureCompensation",
         endpoint="algorithm.stop_exposure_compensation",
         description="Stop exposure compensation algorithm",
         errors=[errors.AlgorithmError, errors.StitcherError]
         )
    """
    @run_on_executor
    def stop_exposure_compensation(self, parameters=None):
        if self.stitcher.algorithm_manager:
            self.stitcher.algorithm_manager.stop_exposure_compensation()

    """
    @api(name="ResetExposureCompensation",
         endpoint="algorithm.reset_exposure_compensation",
         description="Reset exposure compensation",
         errors=[errors.StitcherError]
         )
    """
    @run_on_executor
    def reset_exposure_compensation(self, parameters=None):
        self.project_manager.reset_exposure()

    """
    @api(name="StartMetadataProcessing",
         endpoint="algorithm.start_metadata_processing",
         description="Start exposure & IMU metadata processing",
         errors=[errors.StitcherError]
         )
    """
    @run_on_executor
    def start_metadata_processing(self, parameters=None):
        self.project_manager.start_metadata_processing()

    """
    @api(name="StopMetadataProcessing",
         endpoint="algorithm.stop_metadata_processing",
         description="Stop exposure & IMU metadata processing",
         errors=[errors.StitcherError]
         )
    """
    @run_on_executor
    def stop_metadata_processing(self, parameters=None):
        self.project_manager.stop_metadata_processing()

