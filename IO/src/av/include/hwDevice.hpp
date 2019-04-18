/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#pragma once

#include "mfx/mfxvideo.h"

#if defined(WIN32) || defined(WIN64)
#ifndef D3D_SURFACES_SUPPORT
#define D3D_SURFACES_SUPPORT 1
#endif

#if defined(_WIN32) && !defined(MFX_D3D11_SUPPORT)
#include <sdkddkver.h>
#if (NTDDI_VERSION >= NTDDI_VERSION_FROM_WIN32_WINNT2(0x0602))  // >= _WIN32_WINNT_WIN8
#define MFX_D3D11_SUPPORT 1                                     // Enable D3D11 support if SDK allows
#else
#define MFX_D3D11_SUPPORT 0
#endif
#endif  // #if defined(WIN32) && !defined(MFX_D3D11_SUPPORT)
#endif  // #if defined(WIN32) || defined(WIN64)

#define MSDK_ZERO_MEMORY(VAR) \
  { memset(&VAR, 0, sizeof(VAR)); }
#define MSDK_MEMCPY_VAR(dstVarName, src, count) memcpy_s(&(dstVarName), sizeof(dstVarName), (src), (count))
#define MSDK_SAFE_RELEASE(X) \
  {                          \
    if (X) {                 \
      X->Release();          \
      X = NULL;              \
    }                        \
  }
#define MSDK_CHECK_RESULT(P, X, ERR) \
  {                                  \
    if ((X) > (P)) {                 \
      MSDK_PRINT_RET_MSG(ERR);       \
      return ERR;                    \
    }                                \
  }
#define MSDK_CHECK_POINTER(P, ...) \
  {                                \
    if (!(P)) {                    \
      return __VA_ARGS__;          \
    }                              \
  }
#define MSDK_ARRAY_LEN(value) (sizeof(value) / sizeof(value[0]))

enum {
  MFX_HANDLE_GFXS3DCONTROL = 0x100, /* A handle to the IGFXS3DControl instance */
  MFX_HANDLE_DEVICEWINDOW = 0x101   /* A handle to the render window */
};                                  // mfxHandleType

/// Base class for hw device
class CHWDevice {
 public:
  virtual ~CHWDevice() {}
  /** Initializes device for requested processing.
  @param[in] hWindow Window handle to bundle device to.
  @param[in] nViews Number of views to process.
  @param[in] nAdapterNum Number of adapter to use
  */
  virtual mfxStatus Init(mfxHDL hWindow, mfxU16 nViews, mfxU32 nAdapterNum) = 0;
  /// Reset device.
  virtual mfxStatus Reset() = 0;
  /// Get handle can be used for MFX session SetHandle calls
  virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL *pHdl) = 0;
  /** Set handle.
  Particular device implementation may require other objects to operate.
  */
  virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl) = 0;
  virtual mfxStatus RenderFrame(mfxFrameSurface1 *pSurface, mfxFrameAllocator *pmfxAlloc) = 0;
  virtual void UpdateTitle(double fps) = 0;
  virtual void Close() = 0;
};
