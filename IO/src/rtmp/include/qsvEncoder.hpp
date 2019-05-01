// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <unordered_map>
#include <unordered_set>

#include "videoEncoder.hpp"
#include "amfIncludes.hpp"

extern "C" {
#if defined(_WIN32)
#include "x264/x264.h"
#include <ws2tcpip.h>
#else
#include <unistd.h>
#include <inttypes.h>
#include <x264.h>

#define INFINITE 0xFFFFFFFF
#endif
}

#include "mfxvideo++.h"

#define MFX_IMPL_VIA_MASK(x) (0x0f00 & (x))
#define MSDK_ALIGN16(value) (((value + 15) >> 4) << 4)  // round up to a multiple of 16
#define MSDK_ALIGN32(value) (((value + 31) >> 5) << 5)  // round up to a multiple of 32

#if defined(WIN32) || defined(WIN64)
#define D3D_SURFACES_SUPPORT 1
#else
#define LIBVA_SUPPORT 1
#define LIBVA_X11_SUPPORT 1
#endif

#if defined(_WIN32) && !defined(MFX_D3D11_SUPPORT)
#include <sdkddkver.h>
#if (NTDDI_VERSION >= NTDDI_VERSION_FROM_WIN32_WINNT2(0x0602))  // >= _WIN32_WINNT_WIN8
#define MFX_D3D11_SUPPORT 1                                     // Enable D3D11 support if SDK allows
#else
#define MFX_D3D11_SUPPORT 0
#endif
#endif  // #if defined(WIN32) || defined(WIN64)

#if D3D_SURFACES_SUPPORT
#include "d3d_allocator.h"
#include "d3d11_allocator.h"


#include "d3d_device.h"
#include "d3d11_device.h"
#endif

#ifdef LIBVA_SUPPORT
#include "vaapi_allocator.h"
#include "vaapi_device.h"
#undef Status
#undef None
#endif

#include "sysmem_allocator.h"

namespace VideoStitch {
namespace Output {

enum MemType {
  SYSTEM_MEMORY = 0x00,
  D3D9_MEMORY = 0x01,
  D3D11_MEMORY = 0x02,
};

class QSVEncoder : public VideoEncoder {
 public:
  QSVEncoder()
      : session(nullptr),
        encoder(nullptr),
        hwdev(nullptr),
        allocator(nullptr),
        allocatorParams(nullptr),
        memType(SYSTEM_MEMORY),
        externalAlloc(false),
        encSurfaces(nullptr),
        bitRate(0) {}

  ~QSVEncoder() {
    // release all surfaces
    delete session;
  }

  static Potential<VideoEncoder> createQSVEncoder(const Ptv::Value& config, int width, int height, FrameRate framerate);
  static void supportedEncoders(std::vector<std::string>& codecs);

  mfxStatus init(int width, int height, FrameRate framerate, mfxU16 profile, mfxU16 level, int bitRate,
                 mfxU16 targetUsage, mfxU16 rateControlMethod, mfxU16 numSlice, int gop, MemType memType);

  bool encode(const Frame& videoFrame, std::vector<VideoStitch::IO::DataPacket>& packets) override;

  int getBitRate() const override { return bitRate; }

  bool setBitRate(uint32_t /*maxBitrate*/, uint32_t /*bufferSize*/) { return false; }

  bool dynamicBitrateSupported() const override { return false; }

  char* metadata(char* enc, char* pend) override;

 private:
  static const AVal av_videocodecid;
  static const AVal av_videodatarate;
  static const AVal av_framerate;

  static const std::unordered_map<std::string, int> profileMap;
  static const std::unordered_set<mfxU16> levelSet;
  static const std::unordered_map<std::string, int> rcmodeMap;
  static const std::unordered_map<std::string, MemType> memMap;

  static mfxStatus initSession(MFXVideoSession* session, const MemType memType);
  static mfxU16 profileParser(std::string profile);
  static mfxU16 levelParser(std::string level);
  static mfxU16 rcmodeParser(std::string rcmode);
  static MemType memTypeParser(std::string memtype);

  mfxU16 getFreeSurface(mfxFrameSurface1* surfacesPool, mfxU16 poolSize);

  mfxU16 getFreeSurfaceIndex(mfxFrameSurface1* surfacesPool, mfxU16 poolSize);

  void initEncoderParams(int width, int height, FrameRate framerate, mfxU16 profile, mfxU16 level, int bitRate,
                         mfxU16 targetUsage, mfxU16 rateControlMethod, mfxU16 numSlice, int gop);

  mfxStatus createHWDevice();

  mfxStatus resetDevice();

  void deleteHWDevice() { delete hwdev; }

  mfxStatus createAllocator();

  void deleteAllocator();

  mfxStatus allocFrames();

  void deleteFrames();

  // --------------------------------

  MFXVideoSession* session;
  MFXVideoENCODE* encoder;
  mfxVideoParam params;

  CHWDevice* hwdev;

  MFXFrameAllocator* allocator;
  mfxAllocatorParams* allocatorParams;
  MemType memType;
  bool externalAlloc;  // use memory allocator as external for Media SDK

  mfxFrameSurface1* encSurfaces;  // frames array for encoder input
  mfxFrameAllocRequest req;
  mfxFrameAllocResponse resp;

  mfxBitstream bits;

  int bitRate;
  FrameRate framerate;
};

}  // namespace Output
}  // namespace VideoStitch
