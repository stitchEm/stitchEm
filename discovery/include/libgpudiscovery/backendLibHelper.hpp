// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "config.hpp"
#include "genericDeviceInfo.hpp"

#include <atomic>
#include <cassert>
#include <string>

#ifdef _MSC_VER
#include <windows.h>
#include <delayimp.h>
#endif

namespace VideoStitch {
namespace BackendLibHelper {
/**
 * Select a GPU framework to be used for stitching.
 * On Windows this is done at runtime, loading the selected
 * dll, if it's not loaded yet.
 * On macOs, we need to create/update a symlink pointing to
 * the selected backend, if it's not already pointing to it.
 * We can choose to force the update, to immediately change the
 * symlink, or we can manually do it later. See VSA-6762 for
 * more details about it.
 * @param framework Selected GPU framework
 * @param forceUpdateSymlink if needed, forcely create/modify the library symlink
 * @param needToRestart returns if the application needs to be restarted in order to use the selected framework
 */
VS_DISCOVERY_EXPORT bool selectBackend(const Discovery::Framework& framework, bool* needToRestart = nullptr);
VS_DISCOVERY_EXPORT Discovery::Framework getBestFrameworkAndBackend();
VS_DISCOVERY_EXPORT bool isBackendAvailable(const Discovery::Framework& framework);
#ifdef _MSC_VER
VS_DISCOVERY_EXPORT FARPROC WINAPI vsDelayHook(unsigned dliNotify, PDelayLoadInfo pdli);
#endif
#ifdef __APPLE__
VS_DISCOVERY_EXPORT void forceUpdateSymlink();
#endif
}  // namespace BackendLibHelper
}  // namespace VideoStitch
