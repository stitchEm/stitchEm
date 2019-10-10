// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
// VideoStitch BackendLibHelper
#include <cassert>

#include "libgpudiscovery/backendLibHelper.hpp"
#include "backendLibLoader.hpp"

namespace VideoStitch {
namespace BackendLibHelper {

#ifdef _MSC_VER
// Define hook function for delay load libvideostitch.dll
FARPROC WINAPI vsDelayHook(unsigned dliNotify, PDelayLoadInfo pdli) {
  // check if we have the library to delay load in our map
  if (lstrcmp(pdli->szDll, BackendLibLoader::getDefaultVsLib().c_str()) != 0) {
    return NULL;
  }

  // check if hook function is called for loading the lib
  if (dliNotify != dliNotePreLoadLibrary) {
    return NULL;
  }

  HMODULE dll = BackendLibLoader::getInstance().getBackendHandler();
  if (dll == NULL) {
    assert(false);
    std::cerr << "Error: VideoStitch library handle is invalid" << std::endl;
    return NULL;
  }
  return (FARPROC)dll;
}
#endif

bool isBackendAvailable(const Discovery::Framework& framework) {
  switch (framework) {
    case Discovery::Framework::CUDA:
    case Discovery::Framework::OpenCL:
      return BackendLibLoader::getInstance().isBackendAvailable(framework);
    case Discovery::Framework::Unknown:
    default:
      return false;
  }
}

bool selectBackend(const Discovery::Framework& framework, bool* needToRestart) {
  switch (framework) {
    case Discovery::Framework::CUDA:
    case Discovery::Framework::OpenCL:
      return BackendLibLoader::getInstance().selectBackend(framework, needToRestart);
    case Discovery::Framework::Unknown:
    default:
      if (needToRestart != nullptr) {
        *needToRestart = false;
      }
      return false;
  }
}

Discovery::Framework getBestFrameworkAndBackend() {
  if (Discovery::isFrameworkAvailable(Discovery::Framework::CUDA) && isBackendAvailable(Discovery::Framework::CUDA)) {
    return Discovery::Framework::CUDA;
  } else if (Discovery::isFrameworkAvailable(Discovery::Framework::OpenCL) &&
             isBackendAvailable(Discovery::Framework::OpenCL)) {
    return Discovery::Framework::OpenCL;
  }
  return Discovery::Framework::Unknown;
}
#ifdef __APPLE__
void forceUpdateSymlink() { BackendLibLoader::getInstance().forceUpdateVsSymlink(); }
#endif
}  // namespace BackendLibHelper
}  // namespace VideoStitch
