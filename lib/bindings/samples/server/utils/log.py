import os
import tempfile
import subprocess
import logging
import shutil

from utils.settings_manager import SETTINGS

#tuple in LOG_FILES are (filename, isRotated)
LOG_FILES = {
    "lib videostitch": ("libvideostitch.log", True),
    "python": ("vs_server.log", True),
    "nginx error": ("nginx/nginx_error.log", False),
    "nginx error 2": ("nginx/error.log", False),
    "system/debug": ("debug", True),
    "system/messages": ("messages", False),
    "system/syslog": ("kern.log", False),
    "system/xorg": ("Xorg.0.log", False)
}
VS_LOG_FILE = os.path.join(SETTINGS.log_path, LOG_FILES["lib videostitch"][0])
PYTHON_LOG_FILE = os.path.join(SETTINGS.log_path, LOG_FILES["python"][0])

PACKAGE_FILENAME = "{}_logs.zip"
PACKAGE_PASSWORD = "ICanReadTheLogs"
PACKAGE_COMPRESSION_LEVEL = 5  # 1 (fastest) to 9 (most compressed)
PACKAGE_LOG_MAX_SIZE = 10*1024  # in kB
VERSIONS_FILENAME = "versions.txt"


ROTATE_SIZE = 3

def logrotate(filename):
    try:
        for index in range(9, ROTATE_SIZE, -1):
            new_file_name = filename + "." + str(index)
            if os.path.exists(new_file_name):
                os.remove(new_file_name)
        for index in range(ROTATE_SIZE, 1, -1):
            old_file_name = filename + "." + str(index - 1)
            new_file_name = filename + "." + str(index)
            if os.path.exists(old_file_name):
                os.rename(old_file_name, new_file_name)
        if os.path.exists(filename):
            os.rename(filename, filename + ".1")
    except:
        pass


def set_vs_log(loglevel, rotateAtRuntime=None, message = None):
    """ Sets log level for libvideostitch
    """
    import vs

    loglevels = {
        0: vs.Logger.Error,
        1: vs.Logger.Warning,
        2: vs.Logger.Info,
        3: vs.Logger.Verbose,
        4: vs.Logger.Debug
    }
    if not rotateAtRuntime:
        logrotate(VS_LOG_FILE)
    vs.Logger.setLevel(loglevels[loglevel])
    vs.Logger.setLogFile(VS_LOG_FILE, rotateAtRuntime)
    if message is not None:
        vs.Logger.log(vs.Logger.Info, message)


def set_python_log(loglevel, use_log_file=None, rotateAtRuntime=None):
    """ Redirect the python log to stdio or to a file
    """
    loglevels = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.INFO,
        4: logging.DEBUG
    }
    if use_log_file:
        if rotateAtRuntime:
            # Checks if a log has already happened before configuration
            assert (len(logging.getLogger().handlers) == 0)
            fh = logging.handlers.RotatingFileHandler(PYTHON_LOG_FILE, maxBytes=10*1024*1024, backupCount=4)
            fh.setLevel(loglevels[loglevel])
            logging.getLogger().addHandler(fh)
        else:
            logrotate(PYTHON_LOG_FILE)
            logging.basicConfig(filename=PYTHON_LOG_FILE, level=loglevels[loglevel])
    else:
        logging.basicConfig(level=loglevels[loglevel])

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s][%(threadName)s] %(message)s")
    logging.getLogger().handlers[0].setFormatter(formatter)

def logrotate_previous(base_log_filename):
    return base_log_filename + ".1"

def package_logs(log_file_prefix, versions):
    """ Package all logs for user download
    """
    logs = []
    for logInfo in LOG_FILES.itervalues():
        logs.append(logInfo[0])
        if logInfo[1]:
            logs.append(logrotate_previous(logInfo[0]))

    package_filename = PACKAGE_FILENAME.format(log_file_prefix)

    package_dir = tempfile.mkdtemp()

    # copy logs to temp dir
    copy_and_truncate_logs(logs, package_dir)

    # add a file with version info
    logs.append(create_versions_file(versions, package_dir))

    # create the package
    create_archive(os.path.join(SETTINGS.log_path, package_filename), package_dir, logs)

    return package_filename


def copy_and_truncate_logs(logs, dest_path, truncate=False):
    """
    copy files to dest_path (only last PACKAGE_LOG_MAX_SIZE kilobytes if file is bigger)
    """
    for log_filename in logs:
        file_path = os.path.join(SETTINGS.log_path, log_filename)
        # some of the files might be missing if there was no related error
        if os.path.exists(file_path):
            file_stat = os.stat(file_path)
            tmp_file = os.path.join(dest_path, os.path.basename(log_filename))
            if truncate:
                with open(file_path, 'r') as log_file, \
                        open(tmp_file, 'w') as trunc_file:
                    file_seek = max(file_stat.st_size / 1024 - PACKAGE_LOG_MAX_SIZE + 1, 0)
                    log_file.seek(file_seek * 1024)
                    trunc_file.write(log_file.read())
            else:
                shutil.copyfile(file_path, tmp_file)


def create_versions_file(versions, dest_path):
    version_path = os.path.join(dest_path, VERSIONS_FILENAME)
    with open(version_path, 'w') as version_file:
        for (version_key, version_value) in versions:
            version_file.write(version_key + " : " + version_value + "\r\n")
    return VERSIONS_FILENAME


def create_archive(archive_path, tmp_dir, files):
    # remove an eventual previous archive
    if os.path.exists(archive_path):
        os.remove(archive_path)

    command_line = ['7z', 'a', '-p{}'.format(PACKAGE_PASSWORD), '-y', archive_path]
    for file_name in files:
        file_path = os.path.join(tmp_dir, os.path.basename(file_name))
        if os.path.exists(file_path):
            command_line.append(file_path)
    try:
        subprocess.check_call(command_line)
    except:
        raise Exception("7zip command line failed")
    finally:
        # clean files
        shutil.rmtree(tmp_dir)
