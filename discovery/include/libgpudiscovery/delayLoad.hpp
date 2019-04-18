// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backendLibHelper.hpp"

#ifdef _MSC_VER
#include "delayimp.h"
#include "winerror.h"
#include "delayimp.h"
#endif  // _MSC_VER

// Define hook function for delay load libvideostitch.dll
#ifdef _MSC_VER
#define SET_DELAY_LOAD_HOOK                                             \
  FARPROC WINAPI delayHook(unsigned dliNotify, PDelayLoadInfo pdli) {   \
    return VideoStitch::BackendLibHelper::vsDelayHook(dliNotify, pdli); \
  }                                                                     \
  PfnDliHook __pfnDliNotifyHook2 = delayHook;  // This should be at global scope
#endif                                         // _MSC_VER
