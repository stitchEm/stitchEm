// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/dirutils.hpp"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string.h>
#include <iostream>
#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().

#if defined(_MSC_VER)
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <strsafe.h>
#include <algorithm>
#include <direct.h>
#include <io.h>  // For access().
#include <Lmcons.h>
#else
#include <dirent.h>
#endif

namespace VideoStitch {
namespace Testing {

bool createDirectory(const std::string& absolutePath) {
  int err = 0;
#if defined(_MSC_VER)
  err = _mkdir(absolutePath.c_str());  // can be used on Windows
#else
  err = mkdir(absolutePath.c_str(), 0700);  // can be used on non-Windows
#endif
  return err == 0;
}

Status deleteFile(const std::string& fileName) {
  int ret = remove(fileName.c_str());
  if (ret != 0) {
    return {Origin::Output, ErrType::RuntimeError,
            "Error deleting file " + fileName + ", " + std::string(strerror(errno))};
  }
  return Status::OK();
}

// delete directory, included files and subfolders
Status deleteDir(const std::string& directory) {
  // check if directory exists
  if (!Util::directoryExists(directory)) {
    return Status::OK();
  }
#if defined(_MSC_VER)
  WIN32_FIND_DATA file_data;
  HANDLE hFind = INVALID_HANDLE_VALUE;
  std::string dirWin = directory;
  std::replace(dirWin.begin(), dirWin.end(), '/', '\\');
  dirWin += '\\';

  // delete all files and subfolders
  if ((hFind = FindFirstFile((dirWin + "*.*").c_str(), &file_data)) != INVALID_HANDLE_VALUE) {
    do {
      const std::string fileName = file_data.cFileName;
      const std::string fullFileName = dirWin + fileName;
      const bool isDirectory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

      if (fileName.compare(".") == 0 || fileName.compare("..") == 0) {
        continue;
      }

      // check if it is a directory
      if (isDirectory) {
        // delete directory
        std::cout << "removing  " << std::string(file_data.cFileName) << "  <DIR>" << std::endl;
        FAIL_CAUSE(deleteDir(fullFileName), Origin::Output, ErrType::RuntimeError, "Error deleting " + fullFileName);
      } else {
        // delete file
        std::cout << "removing  " << fullFileName << std::endl;
        FAIL_RETURN(deleteFile(fullFileName));
      }

    } while (FindNextFile(hFind, &file_data));

    FindClose(hFind);
  }
  // finally delete directory
  if (RemoveDirectory(dirWin.c_str()) == FALSE) {
    return {Origin::Output, ErrType::RuntimeError, "Error removing directory " + dirWin};
  }
#else
  DIR* dir;
  struct dirent* ent;
  struct stat st;

  dir = opendir(directory.c_str());
  if (dir != NULL) {
    while (dir != NULL && (ent = readdir(dir)) != NULL) {
      const std::string file_name = ent->d_name;
      const std::string full_file_name = directory + "/" + file_name;

      if (file_name.compare(".") == 0 || file_name.compare("..") == 0) {
        continue;
      }

      if (stat(full_file_name.c_str(), &st) == -1) {
        continue;
      }

      const bool is_directory = (st.st_mode & S_IFDIR) != 0;

      if (is_directory) {
        // delete directory
        std::cout << "removing  " << std::string(full_file_name) << "  <DIR>" << std::endl;
        FAIL_CAUSE(deleteDir(full_file_name), Origin::Output, ErrType::RuntimeError,
                   "Error deleting " + full_file_name);
      } else {
        // delete file
        std::cout << "removing  " << full_file_name << std::endl;
        FAIL_RETURN(deleteFile(full_file_name));
      }
    }
    closedir(dir);
  }
  int rmRet = -1;
  const int MAX_RETRY = 5;
  int retry = 0;

  do {
    rmRet = rmdir(directory.c_str());
  } while (rmRet != 0 && retry++ < MAX_RETRY);

  if (rmRet != 0) {
    return {Origin::Output, ErrType::RuntimeError,
            "Error removing directory " + directory + ", " + std::string(strerror(errno))};
  }
#endif
  return Status::OK();
}

void testGenericDataFolder() {
  Status ret;

  // get default VideoStitch data path, creating it if not yet created
  PotentialValue<std::string> potPath = Util::getVSDataLocation();
  ENSURE(potPath.ok(), potPath.ok() ? "" : potPath.status().getErrorMessage().c_str());

  // make sub dir
  std::string testFolder = potPath.value() + "/testFolder";
  if (!Util::directoryExists(testFolder)) {
    ENSURE(createDirectory(testFolder), "Error creating default VideoStitch data test subfolder");
  }

  // create test file
  const std::string& testFilePath = testFolder + "/test.txt";
  std::ofstream ofs;
  ofs.open(testFilePath.c_str());
  ENSURE(ofs.is_open(), "error creating a file in default VideoStitch data location");
  ofs << "Writing some data\n";
  ofs.close();

  // load test file
  std::ifstream ifs;
  ifs.open(testFilePath.c_str());
  ENSURE(ifs.is_open(), "error loading test file in default VideoStitch data location");
  ifs.close();

  // delete test file
  const int err = remove(testFilePath.c_str());
  ENSURE(err == 0, "error deleting test file in default VideoStitch data location");

  // verify it has been deleted
  struct stat buffer;
  ENSURE(stat(testFilePath.c_str(), &buffer) != 0);

  // delete VideoStitch data folder
  ret = deleteDir(potPath.value());
  ENSURE(ret.ok(), ret.ok() ? "" : ret.getErrorMessage().c_str());
}

void testGenericCacheFolder() {
  Status ret;

  // get default VideoStitch cache path, creating it if not yet created
  PotentialValue<std::string> potPath = Util::getVSCacheLocation();
  ENSURE(potPath.ok(), potPath.ok() ? "" : potPath.status().getErrorMessage().c_str());

  // make sub dir
  std::string testFolder = potPath.value() + "/testFolder";
  if (!Util::directoryExists(testFolder)) {
    ENSURE(createDirectory(testFolder), "Error creating cache test subfolder");
  }

  // create test file
  const std::string& testFilePath = testFolder + "/test.txt";
  std::ofstream ofs;
  ofs.open(testFilePath.c_str());
  ENSURE(ofs.is_open(), "error creating a file in default VideoStitch cache location");
  ofs << "Writing some data\n";
  ofs.close();

  // load test file
  std::ifstream ifs;
  ifs.open(testFilePath.c_str());
  ENSURE(ifs.is_open(), "error loading test file in default VideoStitch cache location");
  ifs.close();

  // delete test file
  const int err = remove(testFilePath.c_str());
  ENSURE(err == 0, "error deleting test file in default VideoStitch cache location");

  // verify it has been deleted
  struct stat buffer;
  ENSURE(stat(testFilePath.c_str(), &buffer) != 0);

  // delete VideoStitch cache folder
  ret = deleteDir(potPath.value());
  ENSURE(ret.ok(), ret.ok() ? "" : ret.getErrorMessage().c_str());
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::testGenericDataFolder();
  VideoStitch::Testing::testGenericCacheFolder();
  return 0;
}
