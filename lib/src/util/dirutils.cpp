// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/dirutils.hpp"

#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().

#ifdef _MSC_VER
#include <algorithm>
#include <direct.h>
#include <io.h>  // For access().
#include <windows.h>
#include <Lmcons.h>
#else
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <cstdlib>
#endif

namespace VideoStitch {
namespace Util {

std::mutex mutexCreateDirectory;
const std::string VIDEOSTITCH_SUB_FOLDER = "VideoStitch";
const std::string CACHE_SUB_FOLDER = "cache";

#ifdef _MSC_VER
const std::string HOME_VAR = "USERPROFILE";
#else
const std::string HOME_VAR = "HOME";
#endif

bool directoryExists(const std::string& absolutePath) {
#ifdef _MSC_VER
  if (_access(absolutePath.c_str(), 0) == 0) {
#endif
    struct stat status;
    return stat(absolutePath.c_str(), &status) == 0 && (status.st_mode & S_IFDIR) != 0;
#ifdef _MSC_VER
  }
  return false;
#endif
}

bool createDirectory(const std::string& absolutePath) {
  int err = 0;
  mutexCreateDirectory.lock();
#if defined(_MSC_VER)
  err = _mkdir(absolutePath.c_str());  // can be used on Windows
#elif defined(__ANDROID__)
  std::string command = "mkdir -p " + absolutePath;
  err = system(command.c_str());  // create subdirectories
#else
  err = mkdir(absolutePath.c_str(), 0700);  // can be used on non-Windows
#endif
  mutexCreateDirectory.unlock();
  return err == 0;
}

std::string getHomeLocation() {
  char const* homeCh = getenv(HOME_VAR.c_str());
  if (!homeCh) {
    return "";
  }
  std::string home = std::string(homeCh);
#if defined(_MSC_VER)
  std::replace(home.begin(), home.end(), '\\', '/');
#endif
  return home;
}

std::string getGenericCacheLocation() {
  const std::string& home = getHomeLocation();
  if (home.empty()) {
    return home;
  }
#if defined(_MSC_VER)
  return home + "/AppData/Local/cache";
#elif defined(__APPLE__)
  return home + "/Library/Caches";
#else  // linux
  return home + "/.cache";
#endif
}

PotentialValue<std::string> getGenericCacheLocation(const std::string& companyName) {
  if (companyName.empty()) {
    return PotentialValue<std::string>({Origin::Output, ErrType::InvalidConfiguration, "Invalid company name"});
  }

  PotentialValue<std::string> companyDataLocation = getGenericDataLocation(companyName);
  if (!companyDataLocation.ok()) {
    return companyDataLocation;
  }

  const std::string& companyCacheLocation = companyDataLocation.value() + "/" + CACHE_SUB_FOLDER;
  if (!directoryExists(companyCacheLocation)) {
    if (!createDirectory(companyCacheLocation)) {
      return PotentialValue<std::string>(
          {Origin::Output, ErrType::RuntimeError,
           "Error creating folder " + companyCacheLocation + ", " + std::string(strerror(errno))});
    }
  }

  return std::string(companyCacheLocation);
}

PotentialValue<std::string> getVSCacheLocation() { return getGenericCacheLocation(VIDEOSTITCH_SUB_FOLDER); }

std::string getGenericDataLocation() {
  const std::string& home = getHomeLocation();
  if (home.empty()) {
    return "";
  }
#if defined(_MSC_VER)
  return home + "/AppData/Local";
#elif defined(__APPLE__)
  return home + "/Library/Application Support";
#else  // linux
  return home + "/.local/share";
#endif
}

PotentialValue<std::string> getGenericDataLocation(const std::string& companyName) {
  if (companyName.empty()) {
    return PotentialValue<std::string>({Origin::Output, ErrType::RuntimeError, "Invalid company name"});
  }

  const std::string& genericDataLocation = getGenericDataLocation();

  if (genericDataLocation.empty()) {
    return PotentialValue<std::string>({Origin::Output, ErrType::RuntimeError, "Invalid generic data location"});
  }

  const std::string& companyDataLocation = genericDataLocation + "/" + companyName;

  if (!directoryExists(companyDataLocation)) {
    if (!createDirectory(companyDataLocation)) {
      return PotentialValue<std::string>(
          {Origin::Output, ErrType::RuntimeError,
           "Error creating folder " + companyDataLocation + ", " + std::string(strerror(errno))});
    }
  }

  return std::string(companyDataLocation);
}

PotentialValue<std::string> getVSDataLocation() { return getGenericDataLocation(VIDEOSTITCH_SUB_FOLDER); }

}  // namespace Util
}  // namespace VideoStitch
