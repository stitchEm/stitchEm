// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
// VideoStitch BackendLibLoader
#pragma once

#include "libgpudiscovery/config.hpp"
#include "libgpudiscovery/genericDeviceInfo.hpp"

#include <atomic>
#include <string>

#ifdef _MSC_VER
#include <windows.h>
#endif  // _MSC_VER

namespace VideoStitch {
namespace BackendLibHelper {
class BackendLibLoader {
 public:
#ifdef __APPLE__
  void forceUpdateVsSymlink();
#endif
  bool isBackendAvailable(const Discovery::Framework& framework);

  bool selectBackend(const Discovery::Framework& framework, bool* needToRestart);

#ifdef _MSC_VER
  HMODULE getBackendHandler() const;
#endif
  static BackendLibLoader& getInstance();
  const static std::string getDefaultVsLib();

 private:
  BackendLibLoader();

#ifdef _MSC_VER
#ifdef DELAY_LOAD_ENABLED
  bool loadDll();
#endif  // DELAY_LOAD_ENABLED
  std::atomic<HMODULE> backendDllHandler;
#elif defined(__APPLE__)
  bool updateVsSymlink();
#endif
  std::atomic<Discovery::Framework> currentVsFramework;
};
}  // namespace BackendLibHelper
}  // namespace VideoStitch
