/* ****************************************************************************** *\

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013 Intel Corporation. All Rights Reserved.

\* ****************************************************************************** */

#include "hwDevice.hpp"

#include "vaapiUtils.hpp"

#include <va/va_drm.h>

CHWDevice* CreateVAAPIDevice(void);

class CLibVA {
 public:
  virtual ~CLibVA(void){};

  VADisplay GetVADisplay(void) { return m_va_dpy; }

 protected:
  CLibVA(void) : m_va_dpy(nullptr) {}
  VADisplay m_va_dpy;
};

class DRMLibVA : public CLibVA {
 public:
  DRMLibVA(void);
  virtual ~DRMLibVA(void);

 protected:
  int m_fd;
};

/** VAAPI DRM implementation. */
class CVAAPIDeviceDRM : public CHWDevice {
 public:
  CVAAPIDeviceDRM() {}
  virtual ~CVAAPIDeviceDRM(void) {}

  virtual mfxStatus Init(mfxHDL hWindow, mfxU16 nViews, mfxU32 nAdapterNum) { return MFX_ERR_NONE; }
  virtual mfxStatus Reset(void) { return MFX_ERR_NONE; }
  virtual void Close(void) {}

  virtual mfxStatus SetHandle(mfxHandleType type, mfxHDL hdl) { return MFX_ERR_UNSUPPORTED; }
  virtual mfxStatus GetHandle(mfxHandleType type, mfxHDL* pHdl) {
    if ((MFX_HANDLE_VA_DISPLAY == type) && (nullptr != pHdl)) {
      *pHdl = m_DRMLibVA.GetVADisplay();

      return MFX_ERR_NONE;
    }
    return MFX_ERR_UNSUPPORTED;
  }

  virtual mfxStatus RenderFrame(mfxFrameSurface1* pSurface, mfxFrameAllocator* pmfxAlloc) { return MFX_ERR_NONE; }
  virtual void UpdateTitle(double fps) {}

 protected:
  DRMLibVA m_DRMLibVA;
};

CLibVA* CreateLibVA(void);
