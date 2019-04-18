import sys
from optparse import OptionParser

from API import schema
from defaults import VAR_LOG_PATH


def dump_schema(option, opt, value, parser):
    schema.dump_schema(parser.values.js2pojo_dir)
    sys.exit(0)


def parse_args():
    # Note please define default values for the parameters in the defaults.py
    usage = "usage : %prog [options]"
    parser = OptionParser(usage)

    parser.add_option("-d",
                      "--lib-path",
                      type="string",
                      dest="lib_path",
                      help="points to library directory")

    parser.add_option("-v",
                      "--verbose",
                      action="store_true",
                      dest="verbose",
                      help="verbose mode")

    parser.add_option("-p",
                      "--port",
                      type="int",
                      dest="port",
                      help="port of the bindings server (default 8877)")

    parser.add_option("-l",
                      "--loglevel",
                      type="int",
                      dest="loglevel",
                      help="log level 0: Error 1: Warning 2: Info 3: Verbose 4: Debug")

    parser.add_option("--pythonlog",
                      action="store_true",
                      dest="python_log_file",
                      help="save the python log into a file")

    parser.add_option("--logrotate",
                      action="store_true",
                      dest="logrotate",
                      help="rotates the python log at run time")

    parser.add_option("--varlog",
                      action="store_const",
                      dest="log_path",
                      const=VAR_LOG_PATH,
                      help="save logs to /var/log")

    parser.add_option("--camera-simulation",
                      action="store_true",
                      dest="camera_simulation",
                      help="start as if camera was connected")

    parser.add_option("--camera-autoconnect",
                      action="store_true",
                      dest="camera_autoconnect",
                      help="start as if camera was connected")

    parser.add_option("--player-dir",
                      type="str",
                      dest="js2pojo_dir",
                      help="path to the Videostitch-Players directory")

    parser.add_option("--dump-schema",
                      action="callback",
                      callback=dump_schema,
                      help="generate schema for java bindings")

    parser.add_option("--profiling_time",
                      type="int",
                      dest="profiling_time",
                      help="Enable profiling for <> seconds")

    parser.add_option("--auto_stream",
                      action="store_true",
                      dest="auto_stream",
                      help="auto start streaming by default")

    parser.add_option("--auto_record",
                      action="store_true",
                      dest="auto_record",
                      help="auto start output record by default")

    parser.add_option("-r",
                      "--resolution",
                      type="string",
                      dest="resolution",
                      help='"4K DCI"|"4K UHD"|"2.8K"|"2K"|"HD"')

    parser.add_option("--procedural",
                      action="store_true",
                      dest="procedural",
                      help='"starts the server in procedural mode"')

    parser.add_option("--with_logo",
                      action="store_true",
                      dest="with_logo",
                      help='"enable logo insertion"')

    parser.add_option("--input-size",
                      type="int",
                      dest="input_size",
                      help='"number of input streams"')

    parser.add_option("--audio",
                      type="string",
                      dest="current_audio_source",
                      help='"noaudio|camera|line-in|[USB card name]"')

    parser.add_option("--disable-audio-lineout",
                      action="store_true",
                      dest="disable_audio_lineout",
                      help='"disable sound output on line-out"')

    parser.add_option("--display",
                      type="string",
                      dest="display",
                      help='"none|window|fullscreen|[screen name]"')

    parser.add_option("--ptv",
                      type="string",
                      dest="ptv",
                      help='"use ptv as a source instead of the run-time configuration"')

    parser.add_option("--force-default",
                      action="store_true",
                      dest="force_default",
                      help='"Overwrite presets with the default ones"')

    parser.add_option("--no-ev-compensation",
                      action="store_false",
                      dest="enable_ev_compensation",
                      help='"Disable exposure compensation algorithm (run by default)"')

    parser.add_option("--no-metadata-processing",
                      action="store_false",
                      dest="enable_metadata_processing",
                      help='"Disable exposure & IMU metadata processing (active by default)"')

    parser.add_option("--ignore-firmware-checks",
                      action="store_true",
                      dest="ignore_firmware_checks",
                      help='"Bypass firmware checks"')

    parser.add_option("--audio_gain_values",
                      type="float",
                      nargs=4,
                      dest="audio_gain_db",
                      help="Audio gain db. Should be set as 4 consecutive space separated numbers.")

    parser.add_option("--disable-output-recovery",
                      action="store_false",
                      dest="output_recovery_enabled",
                      help="Don't recover outputs by default on camera recovery.")

    parser.add_option("--sdcard-fs",
                      type="string",
                      dest="sdcard_filesystem",
                      help='"What filesystem to use to format SDcard. Supported filesystems: [vfat, exfat]"')

    parser.add_option("--recording-safety-margin",
                      type="int",
                      dest="recording_safety_margin",
                      help="How much space should we reserve?")

    parser.add_option("--multiple-outputs",
                      action="store_true",
                      dest="multiple_outputs",
                      help="Enable simultaneaous record and broadcast")

    parser.add_option("--disable-nginx-recording",
                      action="store_true",
                      dest="disable_nginx_recording",
                      help='"Disable use of nginx to record inputs"')

    parser.add_option("--no-update-check",
                      action="store_true",
                      dest="no_update_check",
                      help="Disable check for box software updates")

    parser.add_option("--update-info-url",
                      type="string",
                      dest="update_info_url",
                      help="Set an alternative url for box software updates information file")

    (options, args) = parser.parse_args()

    return options, args
