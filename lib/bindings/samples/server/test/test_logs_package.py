import unittest
import utils.log
import tempfile
import os
import shutil
import zipfile
import re
from utils.settings_manager import SETTINGS

TEST_FILE_CONTENT = "some file content"
LIB_LOG_INDEX = "lib videostitch"
NGINX_LOG_INDEX = "nginx error"

class TestLogsPackage(unittest.TestCase):
    def test_logs_truncating(self):
        logs = [utils.log.LOG_FILES[LIB_LOG_INDEX][0], utils.log.LOG_FILES[NGINX_LOG_INDEX][0]]
        tmp_dir = tempfile.mkdtemp()
        try:
            utils.log.PACKAGE_LOG_MAX_SIZE = 10
            utils.log.copy_and_truncate_logs(logs, tmp_dir, True)
            #check file exists and has a size inferior to truncate size
            for log_filename in logs:
                # nginx file might be missing if there had been no error
                if os.path.exists(os.path.join(SETTINGS.log_path, log_filename)):
                    file_stat = os.stat(os.path.join(tmp_dir, os.path.basename(log_filename)))
                    self.assertLessEqual(file_stat.st_size, utils.log.PACKAGE_LOG_MAX_SIZE * 1024)
        finally:
            # clean files
            shutil.rmtree(tmp_dir)

    def test_archive(self):
        files_dir = tempfile.mkdtemp()
        archive_dir = tempfile.mkdtemp()
        try:
            test_filename = "test"
            test_path = os.path.join(files_dir, test_filename)
            with open(test_path, 'w') as test_file:
                test_file.write(TEST_FILE_CONTENT)
            test_archive = os.path.join(archive_dir, "archive.zip")
            utils.log.create_archive(test_archive, files_dir, [test_filename])
            self.assertFalse(os.path.exists(files_dir))
            with zipfile.ZipFile(test_archive, 'r') as archive:
                self.assertIn(test_filename, archive.namelist())
                with self.assertRaisesRegexp(RuntimeError, re.compile("password required", re.IGNORECASE)):
                    archive.read(test_filename) # try to read without password
                with self.assertRaisesRegexp(RuntimeError, re.compile("bad password",re.IGNORECASE)):
                    archive.read(test_filename, "wrong password")
                self.assertEqual(archive.read(test_filename, utils.log.PACKAGE_PASSWORD), TEST_FILE_CONTENT)
        finally:
            # clean files
            if os.path.exists(files_dir):
                shutil.rmtree(files_dir)
            shutil.rmtree(archive_dir)
