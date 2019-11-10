// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "backendLibLoader.hpp"
#include "libgpudiscovery/fileHelper.hpp"

#include <cassert>
#include <map>

#ifdef _MSC_VER
#include "winerror.h"
const std::string libExt = ".dll";
#else  // _MSC_VER
#include <dlfcn.h>
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
const std::string libExt = ".dylib";
#else
const std::string libExt = ".so";
#endif
#endif  // _MSC_VER

namespace VideoStitch {
namespace BackendLibHelper {

static const std::string defaultVsLib = "libvideostitch" + libExt;
static std::map<Discovery::Framework, std::string> vsLibs = {
    {Discovery::Framework::CUDA, "libvideostitch_cuda" + libExt},
    {Discovery::Framework::OpenCL, "libvideostitch_opencl" + libExt},
    {Discovery::Framework::Unknown, defaultVsLib}};

#ifdef __APPLE__
bool getPathToLibs(std::string& pathToLibs) {
  pathToLibs = "";
  // Get a url to the main bundle
  CFBundleRef mainBundle = CFBundleGetMainBundle();
  if (mainBundle == NULL) {
    std::cerr << "Error retrieving bundle path, no main bundle available" << std::endl;
    return false;
  }

  CFURLRef bundleUrl = CFBundleCopyBundleURL(mainBundle);
  if (bundleUrl == NULL) {
    std::cerr << "Error retrieving bundle path, no main bundle url available" << std::endl;
    return false;
  }

  CFStringRef bundlePath = CFURLCopyFileSystemPath(bundleUrl, kCFURLPOSIXPathStyle);
  CFRelease(bundleUrl);

  // Get the system encoding method
  CFStringEncoding encodingMethod = CFStringGetSystemEncoding();
  // Convert the string reference into a C string
  CFIndex bufLen = CFStringGetMaximumSizeForEncoding(CFStringGetLength(bundlePath), encodingMethod) + 1;
  char bundlePathBuffer[bufLen];
  bool success;
  success = CFStringGetCString(bundlePath, bundlePathBuffer, bufLen, encodingMethod);
  CFRelease(bundlePath);
  if (!success) {
    std::cerr << "Error converting bundle path to string" << std::endl;
    return false;
  }
  const std::string pathToBundle = std::string(bundlePathBuffer);
  pathToLibs = pathToBundle + "/";
  const std::string pathToFramework = "Contents/Frameworks/";
  if (FileHelper::fileExists(pathToLibs + pathToFramework)) {
    // working from a bundle app
    pathToLibs += pathToFramework;
  }
  if (!FileHelper::fileExists(pathToLibs + vsLibs.at(Discovery::Framework::CUDA)) &&
      !FileHelper::fileExists(pathToLibs + vsLibs.at(Discovery::Framework::OpenCL))) {
    // we are in a subfolder
    auto fIndex = pathToLibs.find_last_of("Studio/");
    pathToLibs = pathToLibs.substr(0, fIndex - std::string("Studio/").length() + 1);
  }
  return true;
}
bool readVsSymLink(std::string& targetLib) {
  std::string pathToLibs;
  if (!getPathToLibs(pathToLibs)) {
    return false;
  }
  const std::string pathToSymlink = pathToLibs + defaultVsLib;

  if (FileHelper::fileExists(pathToSymlink)) {
    char buf[1024];
    ssize_t len;
    if ((len = readlink(pathToSymlink.c_str(), buf, sizeof(buf) - 1)) != -1) {
      buf[len] = '\0';
      targetLib = std::string(buf);
      return true;
    }
  }
  return false;
}

bool BackendLibLoader::updateVsSymlink() {
#ifdef SYMLINK_UPDATE_ENABLED
  std::string pathToLibs;
  if (!getPathToLibs(pathToLibs)) {
    return false;
  }
  const std::string pathToSymlink = pathToLibs + defaultVsLib;
  const std::string pathToLib = pathToLibs + vsLibs[currentVsFramework];

  if (FileHelper::fileExists(pathToSymlink)) {
    if (remove(pathToSymlink.c_str()) != 0) {
      std::cerr << "Error removing" << pathToSymlink << std::endl;
      return false;
    }
  }

  if (!FileHelper::fileExists(pathToLib)) {
    std::cerr << "Library " << pathToLib << " not found" << std::endl;
    return false;
  }

  if (symlink(vsLibs[currentVsFramework].c_str(), pathToSymlink.c_str()) != 0) {
    std::cerr << "Error creating symlink to " << pathToSymlink << std::endl;
    return false;
  }
#endif  // SYMLINK_UPDATE_ENABLED
  return true;
}
#endif  // __APPLE__

BackendLibLoader::BackendLibLoader()
    :
#ifdef _MSC_VER
      backendDllHandler(NULL),
#endif
      currentVsFramework(Discovery::Framework::Unknown) {
#ifdef __APPLE__
  // get current lib for apple
  std::string currentLib;
  if (readVsSymLink(currentLib)) {
    if (currentLib.find(vsLibs.at(Discovery::Framework::CUDA)) != std::string::npos) {
      currentVsFramework = Discovery::Framework::CUDA;
    } else if (currentLib.find(vsLibs.at(Discovery::Framework::OpenCL)) != std::string::npos) {
      currentVsFramework = Discovery::Framework::OpenCL;
    }
    std::cout << "Current backend is " << Discovery::getFrameworkName(currentVsFramework) << std::endl;
  }
#endif
}

bool BackendLibLoader::isBackendAvailable(const Discovery::Framework& framework) {
  if (framework == Discovery::Framework::Unknown) {
    return false;
  }
  bool ret = false;
  // try load library
#ifdef _MSC_VER
  HMODULE dllHandle = NULL;
  dllHandle = LoadLibraryA(vsLibs.at(framework).c_str());
  if (dllHandle != NULL) {
    ret = true;
    FreeLibrary(dllHandle);
  }
#elif defined(__APPLE__)
  std::string pathToLibs;
  if (!getPathToLibs(pathToLibs)) {
    return false;
  }
  const std::string pathToLib = pathToLibs + vsLibs[framework];

  if (FileHelper::fileExists(pathToLib)) {
    ret = true;
  }
#else
  const std::string& libName = vsLibs.at(framework);
  void* dllHandle = dlopen(libName.c_str(), RTLD_LOCAL | RTLD_LAZY);
  if (dllHandle != NULL) {
    ret = true;
    dlclose(dllHandle);
  }
#endif
  return ret;
}

bool BackendLibLoader::selectBackend(const Discovery::Framework& framework, bool* needToRestart) {
  if (needToRestart) {
    *needToRestart = false;
  }

#if defined(SYMLINK_UPDATE_ENABLED) || defined(DELAY_LOAD_ENABLED)
  if (framework == Discovery::Framework::Unknown) {
    return false;
  }
  if (currentVsFramework == framework) {
    return true;
  }
  currentVsFramework = framework;

#if _MSC_VER
  if (getBackendHandler() != NULL) {
    if (needToRestart) {
      *needToRestart = true;
    }
    std::cout << "Backend dll handler already loaded" << std::endl;
    return false;
  }
  return loadDll();
#elif defined(__APPLE__)
  if (needToRestart) {
    *needToRestart = true;
  }
  return updateVsSymlink();
#else
  return true;
#endif  // _MSC_VER
#else
  (void)framework;
  return true;
#endif // defined(SYMLINK_UPDATE_ENABLED) || defined(DELAY_LOAD_ENABLED)
}

#ifdef __APPLE__
void BackendLibLoader::forceUpdateVsSymlink() { updateVsSymlink(); }
#endif

BackendLibLoader& BackendLibLoader::getInstance() {
  static BackendLibLoader istance;
  return istance;
}

const std::string BackendLibLoader::getDefaultVsLib() { return defaultVsLib; }

#ifdef _MSC_VER
std::string getLastErrorAsString() {
  // Get the error message, if any.
  DWORD errorMessageID = ::GetLastError();
  if (errorMessageID == 0) {
    return std::string();  // No error message has been recorded
  }

  LPSTR messageBuffer = nullptr;
  size_t size =
      FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                     errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

  std::string message(messageBuffer, size);

  // Free the buffer.
  LocalFree(messageBuffer);

  return message;
}

#ifdef DELAY_LOAD_ENABLED
bool BackendLibLoader::loadDll() {
  if (getBackendHandler() != NULL) {
    std::cout << "Backend dll handler already loaded" << std::endl;
    return false;
  }
  std::cout << "Loading " << vsLibs.at(currentVsFramework) << std::endl;
  backendDllHandler = LoadLibraryA(vsLibs.at(currentVsFramework).c_str());
  if (getBackendHandler() == NULL) {
    std::cerr << "Warning: could not load " << vsLibs.at(currentVsFramework) << std::endl;
    std::cerr << getLastErrorAsString() << std::endl;
    currentVsFramework = Discovery::Framework::Unknown;
    return false;
  }
  return true;
}
#endif  // DELAY_LOAD_ENABLED

HMODULE BackendLibLoader::getBackendHandler() const { return backendDllHandler; }
#endif
}  // namespace BackendLibHelper
}  // namespace VideoStitch
