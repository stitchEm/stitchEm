import os
import uuid

from utils.filesystem import FAT32

SESSION_UUID = str(uuid.uuid4())

FILE_PATH = os.path.realpath(os.path.abspath(__file__))
DIR_PATH = os.path.dirname(FILE_PATH)

USER_PATH = os.path.join(os.sep, "data", "videostitch")
USER_CONFIG_PATH = os.path.join(USER_PATH, 'config')
USER_RECORDINGS_PATH = os.path.join(USER_PATH, "rec")
NGINX_RECORDINGS_PATH = os.path.join(USER_PATH, "recordings")

USER_PRESETS_DIR_PATH = os.path.join(USER_CONFIG_PATH, 'presets')
SYSTEM_PRESETS_DIR_PATH = os.path.join(DIR_PATH, 'config', 'presets')

PRESET_EXT = ".preset"
DEFAULT_PRESET_FILENAME_NOEXT = "default"
DEFAULT_PRESET_FILENAME = "default" + PRESET_EXT
LOCAL_BROADCAST_PRESET = "local-preview-server"

PID_PATH = os.path.join(USER_PATH, "vs_server.pid")

USER_DEFAULT_PTV_PATH = os.path.join(USER_PRESETS_DIR_PATH, 'user-default.ptv')
SYSTEM_DEFAULT_PTV_PATH = os.path.join(SYSTEM_PRESETS_DIR_PATH, 'default.ptv')
FIRMWARE_DIR_PATH = os.path.join(DIR_PATH, "firmware")
RIG_PARAMETERS_FILEPATH = os.path.join(USER_PATH, "camera-rig-preset.json")

VAR_LOG_PATH = os.path.realpath(os.path.abspath("/var/log"))
VERSION_LOG_PATH = os.path.join(VAR_LOG_PATH, "version")


FIRMWARE_EXTENSION = ".fwupd"
CAMERA_SERVICE_NAME = "_vscamera._tcp.local."
CAMERA_DEFAULT_ETH0_IP = "169.254.87.181"
CAMERA_DEFAULT_PORT = "9989"
CAMERA_CONTROL_ENDPOINT = "control"
STREAM_ENDPOINT = "inputs"
FIRMWARE_ENDPOINT = "firmware"

CAMERA_BOOT_DELAY = 3
FIRST_PING = 10000
PING_INTERVAL = 3000
PONG_TIMEOUT = 2000

DISPLAY_BUFFER_SIZE = 16

SDCARD_MOUNTPOINT = "SDCARD"
NETWORK_OUTPUT_INTERFACE = "eth1"

ENCODER_VERSION = "Orah-4i"

DEFAULT_OPTIONS = {
    "current_audio_source" : "camera",
    "current_audio_layout" : "stereo",
    "disable_audio_lineout": False,
    "audio_gain_db": [
        20,
        20,
        20,
        20
    ],
    "audio_delays": {},
    "audio_base_delay": 1.1,
    "auto_record": False,
    "auto_stream": False,
    "camera_autoconnect": False,
    "camera_simulation": False,
    "no_update_check": False,
    "enable_ev_compensation": False,
    "enable_metadata_processing": True,
    "enable_stabilization": False,
    "force_default": False,
    "ignore_firmware_checks": False,
    "input_size": 4,
    "js2pojo_dir": None,
    "last_preset_streaming": None,
    "lib_path": os.path.join(DIR_PATH, "..", "..", "lib", "release"),
    "log_path": VAR_LOG_PATH,
    "loglevel": 1,
    "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
    "output_recovery_enabled": True,
    "port": 8877,
    "procedural": False,
    "profiling_time": 0,
    "ptv": None,
    "python_log_file": False,
    "logrotate" : False,
    "recording_safety_margin": 150,
    "recording_warning_margin": 300,
    "resolution": "4K DCI",
    "save_algorithm_results": False,
    "sdcard_filesystem": FAT32,
    "social_links": {},
    "output_file_index": 0,
    "update_info_url": "http://s3.video-stitch.com/orah/orah4i.release",
    "verbose": False,
    "multiple_outputs": False,
    "disable_nginx_recording": False,
    "display": "fullscreen",
    "with_logo": False
}

# Messages
RESET_TO_DEFAULT_MESSAGE = "reset_to_default_ptv"
