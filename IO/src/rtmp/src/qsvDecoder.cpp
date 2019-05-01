// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videoDecoder.hpp"

#include "mfx_buffering.h"

#include "libvideostitch/logging.hpp"

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

#include <thread>

static const std::string QSVtag("QSV Decoder");

#define SURFACE_RETRY 5

#define MFX_IMPL_VIA_MASK(x) (0x0f00 & (x))
#define MSDK_ALIGN16(value) (((value + 15) >> 4) << 4)  // round up to a multiple of 16
#define MSDK_ALIGN32(value) (((value + 31) >> 5) << 5)  // round up to a multiple of 32
#define IMSDK_CHECK_RESULT(P, X, ERR)                                                                       \
  {                                                                                                        \
    if ((X) > (P)) {                                                                                       \
      Logger::warning(QSVtag) << __FUNCTION__ << " line " << __LINE__ << "return with error code " << ERR; \
      return ERR;                                                                                          \
    }                                                                                                      \
  }

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
#endif

#include "sysmem_allocator.h"


const static mfxU8 start_seq[] = {0, 0, 0, 1};

enum MemType {
  SYSTEM_MEMORY = 0x00,
  D3D9_MEMORY = 0x01,
  D3D11_MEMORY = 0x02,
};

namespace VideoStitch {
namespace Input {

class QSVDecoder : public VideoDecoder, private CBuffering {
 public:
  QSVDecoder() {
    sequenceHeader = false;
    memset(&bitS, 0, sizeof(bitS));
    bitS.MaxLength = 1024 * 1024 * 20;  // as QSV example...
    bitS.Data = new mfxU8[bitS.MaxLength];
    bitS.DataOffset = 0;
    bitS.DataLength = 0;
    lastTimestamp = 0;
    inputTimestamp = 0;
    latency = 0;
  }

  ~QSVDecoder() {
    decoder->Close();
    delete decoder;
    deleteFrames();
    // release all surfaces
    deleteAllocator();
    session->Close();
    delete session;
  }

  mfxStatus init(int width, int height, FrameRate framerate, MemType memType) {
    this->framerate = framerate;
    this->memType = memType;

    session = new MFXVideoSession;

    // we set version to 1.0 and later we will query actual version of the library which will got leaded
    mfxVersion min_version;
    min_version.Major = 1;
    min_version.Minor = 0;
    // try searching on all display adapters
    mfxIMPL impl = MFX_IMPL_HARDWARE_ANY;
    // if d3d11 surfaces are used ask the library to run acceleration through D3D11
    // feature may be unsupported due to OS or MSDK API version
    if (D3D11_MEMORY == memType) {
      impl |= MFX_IMPL_VIA_D3D11;
    }
    mfxStatus sts = session->Init(impl, &min_version);
    // MSDK API version may not support multiple adapters - then try initialize on the default
    if (sts != MFX_ERR_NONE) {
      sts = session->Init((impl & (!MFX_IMPL_HARDWARE_ANY)) | MFX_IMPL_HARDWARE, nullptr);
    }

    if (sts != MFX_ERR_NONE) {
      Logger::error(QSVtag) << "Unable to init session with error " << sts << std::endl;
      return sts;
    }
    mfxVersion ver;
    sts = session->QueryIMPL(&impl);
    if (sts != MFX_ERR_NONE) {
      Logger::error(QSVtag) << "Unable to query implementation with error " << sts << std::endl;
    }

    sts = session->QueryVersion(&ver);
    if (sts != MFX_ERR_NONE) {
      Logger::error(QSVtag) << "Unable to query version with error " << sts << std::endl;
    }

    std::string sImpl =
        (MFX_IMPL_VIA_D3D11 == MFX_IMPL_VIA_MASK(impl)) ? "hw_d3d11" : (MFX_IMPL_HARDWARE & impl) ? "hw" : "sw";
    Logger::info(QSVtag) << "Media SDK impl     " << sImpl << std::endl;
    Logger::info(QSVtag) << "Media SDK version  " << ver.Major << "." << ver.Minor << std::endl;

    // create and init frame allocator
    sts = createAllocator();
    if (sts != MFX_ERR_NONE) {
      Logger::error(QSVtag) << "Unable to create surface allocator with error " << sts << std::endl;
      return sts;
    }
    // parameters for the decoding
    initDecoderParams(width, height, framerate,
                      // bitRate, targetUsage, rateControlMethod,
                      // numSlice,
                      memType);
    decoder = new MFXVideoDECODE(*session);

    return sts;
  }

  void decodeHeader(Span<const unsigned char> pkt, mtime_t, Span<unsigned char>& header) override {
    memcpy(bitS.Data, start_seq, 4);
    uint16_t spsLen = ((uint32_t)(pkt[11]) << 8) + pkt[12];
    memcpy(bitS.Data + 4, pkt.data() + 13, spsLen);

    memcpy(bitS.Data + 4 + spsLen, start_seq, 4);
    uint16_t ppsLen = ((uint32_t)(pkt[13 + spsLen + 1]) << 8) + pkt[13 + spsLen + 2];
    memcpy(bitS.Data + 4 + spsLen + 4, pkt.data() + 13 + spsLen + 3, ppsLen);

    bitS.DataLength = 4 + spsLen + 4 + ppsLen;

    codecConfig = true;
    header = Span<unsigned char>(bitS.Data + bitS.DataOffset, bitS.DataLength);
  }

  bool demux(Span<const unsigned char> pkt, mtime_t timestamp, VideoStitch::IO::Packet& avpkt) override {
    memmove(bitS.Data, bitS.Data + bitS.DataOffset, bitS.DataLength);
    bitS.DataOffset = 0;

    size_t bytesLeft = pkt.size() - 5;
    const unsigned char* ptr = pkt.data() + 5;
    while (bytesLeft) {
      uint32_t dataSize = ((uint32_t)(*(ptr)) << 24) + ((uint32_t)(*(ptr + 1)) << 16) + ((uint32_t)(*(ptr + 2)) << 8) +
                          (uint32_t)(*(ptr + 3));
      if (dataSize > (bytesLeft - 4)) {
        Logger::error(QSVtag) << "Data packet size corrupted : " << dataSize << ",  clipping to " << bytesLeft - 4
                              << std::endl;
        dataSize = (uint32_t)(bytesLeft - 4);
      }
      switch (*(ptr + 4) & 0x1f) {
        case NAL_SLICE:
        case NAL_SLICE_IDR:
          if ((dataSize + bitS.DataLength + 4) > bitS.MaxLength) {
            Logger::error(QSVtag) << "Drop packet of size " << dataSize
                                  << " to avoid Bitstream buffer overflow : " << bitS.DataLength << ",  max to "
                                  << bitS.MaxLength << std::endl;
            break;
          }
          memcpy(bitS.Data + bitS.DataLength, start_seq, 4);
          memcpy(bitS.Data + bitS.DataLength + 4, ptr + 4, dataSize);
          bitS.DataLength += dataSize + 4;
          break;
      }
      bytesLeft -= dataSize + 4;
      ptr += dataSize + 4;
    }

    if (bitS.DataLength == 0) {
      // dropped everything
      Logger::error(QSVtag) << "No data available, dropping RTMP packet " << std::endl;
      return false;
    }
    bitS.TimeStamp =
        mfxU64(90 * (timestamp + ((uint64_t)(pkt[2]) << 16) + ((uint64_t)(pkt[3]) << 8) + ((uint64_t)(pkt[4]))));
    inputTimestamp = bitS.TimeStamp;

    avpkt.data = Span<unsigned char>(bitS.Data + bitS.DataOffset, bitS.DataLength);
    avpkt.pts = (timestamp + ((uint64_t)(pkt[2]) << 16) + ((uint64_t)(pkt[3]) << 8) + ((uint64_t)(pkt[4]))) * 1000;
    avpkt.dts = timestamp * 1000;

    return true;
  }

  bool decodeAsync(VideoStitch::IO::Packet&) override {
    if (!sequenceHeader) {
      // the decoder needs an I-frame after the SPS and PPS in order
      // to be initialized correctly
      mfxStatus sts = decoder->DecodeHeader(&bitS, &params);
      if (sts != MFX_ERR_NONE) {
        Logger::error(QSVtag) << "Unable to decode header for init with error " << sts << std::endl;
      }
      sts = allocFrames();
      if (sts != MFX_ERR_NONE) {
        Logger::error(QSVtag) << "Unable to alloc frame surface with error " << sts << std::endl;
      }
      sts = decoder->Init(&params);
      if (sts != MFX_ERR_NONE) {
        Logger::error(QSVtag) << "Unable to init decoder with error " << sts << std::endl;
      }
      sequenceHeader = true;
    }

    return decode();
  }

  bool decode() {
    mfxStatus sts = MFX_ERR_MORE_DATA;
    bool frameDecoded = false;

    do {
      // get surfaces for decoder
      claimSurfacesForDecoder();

      if (!cleanupIfSurfacesUnavailabe()) {
        Logger::error(QSVtag) << "No surface available, waiting for surfaces " << std::endl;
        return false;
      }

      // decode the bitstream
      mfxFrameSurface1* pOutSurface = nullptr;
      int retries = 3;

      do {
        sts = decoder->DecodeFrameAsync(&bitS, &(m_pCurrentFreeSurface->frame), &pOutSurface,
                                        &(m_pCurrentFreeOutputSurface->syncp));
        if (sts == MFX_WRN_DEVICE_BUSY) {
          retries--;
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
          break;
        }
      } while (retries != 0);

      if (sts > MFX_ERR_NONE) {
        // Get rid of warnings
        if (m_pCurrentFreeOutputSurface->syncp) {
          sts = MFX_ERR_NONE;
        } else {
          sts = MFX_ERR_MORE_DATA;
        }
      }

      if (!(sts == MFX_ERR_NONE || sts == MFX_ERR_MORE_DATA || sts == MFX_ERR_MORE_SURFACE)) {
        Logger::error(QSVtag) << "Potential error on DecodeFrameAsync with value " << sts << std::endl;
      }

      cleanupAfterDecodeFrameAsync(sts, pOutSurface);

      if (MFX_ERR_NONE == sts) {
        frameDecoded = true;
      }

      // We loop until the decoder asks for more data or surface
    } while (sts == MFX_ERR_NONE || sts == MFX_ERR_MORE_SURFACE);

    return frameDecoded;
  }

  size_t flush() {
    size_t flushed_count = 0;

    do {
      if (decode()) {
        flushed_count++;
      } else {
        break;
      }
    } while (true);

    Logger::info(QSVtag) << "Flushed " << flushed_count << std::endl;

    return flushed_count;
  }

  bool synchronize(mtime_t& date, VideoPtr& videoFrame) override {
    msdkOutputSurface* outputSurface = m_OutputSurfacesPool.GetSurface();
    if (!outputSurface) {
      return false;  // no picture yet
    }

    mfxStatus sts = session->SyncOperation(outputSurface->syncp, INFINITE);
    if (sts != MFX_ERR_NONE) {
      Logger::error(QSVtag) << "Unable to sync/get surface at decoder output with error " << sts << std::endl;
    }

    mfxFrameSurface1 surf = outputSurface->surface->frame;

    // the unit used by the MFX framework is 90 Khz, we want 1 Mhz
    date = mtime_t(surf.Data.TimeStamp * 1000 / 90);
    videoFrame = (VideoPtr)outputSurface;

    // Check latency
    mtime_t currentLatency = inputTimestamp - surf.Data.TimeStamp;
    if (latency < currentLatency) {
      latency = currentLatency;
      Logger::verbose(QSVtag) << "Video latency increased to " << latency / 90 << " ms" << std::endl;
    }

    if (lastTimestamp >= date) {
      Logger::warning(QSVtag) << "Timestamp error " << lastTimestamp << " " << date << std::endl;
    }

    lastTimestamp = date;

    return true;
  }

  void stop() {}

  void copyFrame(unsigned char* videoFrame, mtime_t&, VideoPtr videoSurface) override {
    msdkOutputSurface* outputSurface = (msdkOutputSurface*)videoSurface;
    mfxFrameSurface1 surf = outputSurface->surface->frame;

    if (externalAlloc) {
      // if we share allocator with Media SDK we need to call Lock to access surface data
      allocator->Lock(allocator->pthis, surf.Data.MemId, &(surf.Data));
    }

    // get frame content from decoding
    uint8_t* out_ptr = videoFrame;
    mfxU16 w, h;
    mfxU16 pitch = surf.Data.Pitch;
    if (surf.Info.CropH > 0 && surf.Info.CropW > 0) {
      w = surf.Info.CropW;
      h = surf.Info.CropH;
    } else {
      w = surf.Info.Width;
      h = surf.Info.Height;
    }

    // luminance plane
    const mfxU8* ptrY = surf.Data.Y + surf.Info.CropX + surf.Info.CropY * surf.Data.Pitch;
    for (auto i = 0; i < h; i++) {
      memcpy(out_ptr, ptrY + i * pitch, w);
      out_ptr += w;
    }
    // chroma planes
    h /= 2;
    const mfxU8* ptrUV = surf.Data.UV + surf.Info.CropX + (surf.Info.CropY / 2) * pitch;
    for (auto i = 0; i < h; i++) {
      memcpy(out_ptr, ptrUV + i * pitch, w);
      out_ptr += w;
    }

    if (externalAlloc) {
      allocator->Unlock(allocator->pthis, surf.Data.MemId, &(surf.Data));
    }
  }

  void releaseFrame(VideoPtr videoSurface) override { ReturnSurfaceToBuffers((msdkOutputSurface*)videoSurface); }

 private:
  static const std::chrono::milliseconds GET_SURFACE_TIMEOUT;

  void initDecoderParams(int width, int height, FrameRate framerate,
                         // int bitRate,
                         // mfxU16 targetUsage,
                         // mfxU16 rateControlMethod,
                         // mfxU16 numSlice,
                         MemType memType) {
    memset(&params, 0, sizeof(params));

    params.mfx.CodecId = MFX_CODEC_AVC;
    params.mfx.FrameInfo.FrameRateExtN = framerate.num;
    params.mfx.FrameInfo.FrameRateExtD = framerate.den;
    params.AsyncDepth = 10;

    // specify memory type
    if (memType == D3D9_MEMORY || memType == D3D11_MEMORY) {
      params.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    } else {
      params.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    }

    // frame info parameters
    params.mfx.FrameInfo.FourCC = MFX_FOURCC_YV12;
    params.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
    params.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;

    // set frame size and crops
    // width must be a multiple of 32
    // height must be a multiple of 32 in case of frame picture
    params.mfx.FrameInfo.Width = MSDK_ALIGN32(width);
    params.mfx.FrameInfo.Height = MSDK_ALIGN32(height);

    params.mfx.FrameInfo.CropX = 0;
    params.mfx.FrameInfo.CropY = 0;
    params.mfx.FrameInfo.CropW = width;
    params.mfx.FrameInfo.CropH = height;
  }

  void cleanupAfterDecodeFrameAsync(const mfxStatus sts, mfxFrameSurface1* pOutSurface) {
    if ((MFX_ERR_NONE == sts) || (MFX_ERR_MORE_DATA == sts) || (MFX_ERR_MORE_SURFACE == sts)) {
      m_UsedSurfacesPool.AddSurface(m_pCurrentFreeSurface);
      m_pCurrentFreeSurface = nullptr;
    }
    if (MFX_ERR_NONE == sts) {
      // Comments were added some time later
      // As I understand it this is valid because DecodeFrameAsync assigns to pOutSurface a value of a surface that we
      // gave it as an input surface And we have initially allocated it as a part of msdkFrameSurface structure. (It's
      // important to have the mfxFrameSurface1 as a first member of a structure)

      msdkFrameSurface* surface = (msdkFrameSurface*)(pOutSurface);  // just a cast
      ++surface->render_lock;

      m_pCurrentFreeOutputSurface->surface = surface;

      m_OutputSurfacesPool.AddSurface(m_pCurrentFreeOutputSurface);

      m_pCurrentFreeOutputSurface = nullptr;
    }
  }

  void claimSurfacesForDecoder(const int retryCount = SURFACE_RETRY) {
    uint8_t counter = 0;
    do {
      SyncFrameSurfaces();
      if (!m_pCurrentFreeSurface) {
        m_pCurrentFreeSurface = m_FreeSurfacesPool.GetSurface();
      }
      if (!m_pCurrentFreeOutputSurface) {
        m_pCurrentFreeOutputSurface = GetFreeOutputSurface();
      }
      if (!m_pCurrentFreeSurface || !m_pCurrentFreeOutputSurface) {
        // we stuck with no free surface available, now throttling...
        std::this_thread::sleep_for(GET_SURFACE_TIMEOUT);
        Logger::error(QSVtag) << "No surface available, throttling " << std::endl;
      }
      counter++;
    } while ((counter < retryCount) && (!m_pCurrentFreeSurface || !m_pCurrentFreeOutputSurface));
  }

  bool cleanupIfSurfacesUnavailabe() {
    if (!m_pCurrentFreeSurface || !m_pCurrentFreeOutputSurface) {
      // we don't have any surface after 50ms let's drop the packet
      if (m_pCurrentFreeSurface) {
        m_FreeSurfacesPool.AddSurface(m_pCurrentFreeSurface);
        m_pCurrentFreeSurface = nullptr;
      } else if (m_pCurrentFreeOutputSurface) {
        AddFreeOutputSurface(m_pCurrentFreeOutputSurface);
        m_pCurrentFreeOutputSurface = nullptr;
      }
      return false;
    }
    return true;
  }

  mfxStatus createHWDevice() {
    mfxU32 adapterNum = 0;
    mfxIMPL impl = MFX_IMPL_SOFTWARE;  // default in case no HW IMPL is found
    // we don't care for error codes in further code; if something goes wrong we fall back to the default adapter
    if (session) {
      session->QueryIMPL(&impl);
    } else {
      // an auxiliary session, internal for this function
      mfxSession auxSession;
      memset(&auxSession, 0, sizeof(auxSession));

      mfxVersion ver = {{1, 1}};  // minimum API version which supports multiple devices
      mfxStatus sts = MFXInit(MFX_IMPL_HARDWARE_ANY, &ver, &auxSession);
      if (sts != MFX_ERR_NONE) {
        Logger::error(QSVtag) << "Unable to init mfx dispatcher lib with error " << sts << std::endl;
      }

      sts = MFXQueryIMPL(auxSession, &impl);
      if (sts != MFX_ERR_NONE) {
        Logger::error(QSVtag) << "Unable to get mfx dispatcher implementation with error " << sts << std::endl;
      }

      sts = MFXClose(auxSession);
      if (sts != MFX_ERR_NONE) {
        Logger::error(QSVtag) << "Unable to init decoder with error " << sts << std::endl;
      }
    }

    // extract the base implementation type
    mfxIMPL baseImpl = MFX_IMPL_BASETYPE(impl);
    const struct {
      // actual implementation
      mfxIMPL impl;
      // adapter's number
      mfxU32 adapterID;
    } implTypes[] = {{MFX_IMPL_HARDWARE, 0},
                     {MFX_IMPL_SOFTWARE, 0},
                     {MFX_IMPL_HARDWARE2, 1},
                     {MFX_IMPL_HARDWARE3, 2},
                     {MFX_IMPL_HARDWARE4, 3}};

    // get corresponding adapter number
    for (mfxU8 i = 0; i < sizeof(implTypes) / sizeof(*implTypes); i++) {
      if (implTypes[i].impl == baseImpl) {
        adapterNum = implTypes[i].adapterID;
        break;
      }
    }

    mfxStatus sts = MFX_ERR_NONE;
#if D3D_SURFACES_SUPPORT
    // POINT point = { 0, 0 };
    // HWND window = WindowFromPoint(point);

#if MFX_D3D11_SUPPORT
    if (D3D11_MEMORY == memType) {
      hwdev = new CD3D11Device();
    } else
#endif  // #if MFX_D3D11_SUPPORT
      hwdev = new CD3D9Device();

    if (hwdev == nullptr) {
      return MFX_ERR_MEMORY_ALLOC;
    }

    sts = hwdev->Init(
        nullptr, 0,
        adapterNum);  // XXX TODO FIXME https://software.intel.com/en-us/forums/intel-media-sdk/topic/599935#node-599935

#elif LIBVA_SUPPORT

    hwdev = CreateVAAPIDevice();
    if (hwdev == nullptr) {
      return MFX_ERR_MEMORY_ALLOC;
    }

    sts = hwdev->Init(nullptr, 0, adapterNum);
    if (sts != MFX_ERR_NONE) {
      Logger::error(QSVtag) << "Unable to init HW decoder with error " << sts << std::endl;
    }
#endif
    return sts;
  }

  mfxStatus resetDevice() {
    if (D3D9_MEMORY == memType || D3D11_MEMORY == memType) {
      return hwdev->Reset();
    }
    return MFX_ERR_NONE;
  }

  void deleteHWDevice() { delete hwdev; }

  mfxStatus createAllocator() {
    mfxStatus sts = MFX_ERR_NONE;

    if (SYSTEM_MEMORY != memType) {
#if D3D_SURFACES_SUPPORT
      sts = createHWDevice();
      IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

      mfxHDL hdl = NULL;
      mfxHandleType hdl_t =
#if MFX_D3D11_SUPPORT
          D3D11_MEMORY == memType ? MFX_HANDLE_D3D11_DEVICE :
#endif  // #if MFX_D3D11_SUPPORT
                                  MFX_HANDLE_D3D9_DEVICE_MANAGER;

      sts = hwdev->GetHandle(hdl_t, &hdl);
      IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

      // handle is needed for HW library only
      mfxIMPL impl = 0;
      session->QueryIMPL(&impl);
      if (impl != MFX_IMPL_SOFTWARE) {
        sts = session->SetHandle(hdl_t, hdl);
        IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
      }

      // create D3D allocator
#if MFX_D3D11_SUPPORT
      if (D3D11_MEMORY == memType) {
        allocator = new D3D11FrameAllocator;
        MSDK_CHECK_POINTER(allocator, MFX_ERR_MEMORY_ALLOC);

        D3D11AllocatorParams* pd3dAllocParams = new D3D11AllocatorParams;
        MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
        pd3dAllocParams->pDevice = reinterpret_cast<ID3D11Device*>(hdl);

        allocatorParams = pd3dAllocParams;
      } else
#endif  // #if MFX_D3D11_SUPPORT
      {
        allocator = new D3DFrameAllocator;
        MSDK_CHECK_POINTER(allocator, MFX_ERR_MEMORY_ALLOC);

        D3DAllocatorParams* pd3dAllocParams = new D3DAllocatorParams;
        MSDK_CHECK_POINTER(pd3dAllocParams, MFX_ERR_MEMORY_ALLOC);
        pd3dAllocParams->pManager = reinterpret_cast<IDirect3DDeviceManager9*>(hdl);

        allocatorParams = pd3dAllocParams;
      }

      /* In case of video memory we must provide MediaSDK with external allocator
      thus we demonstrate "external allocator" usage model.
      Call SetAllocator to pass allocator to Media SDK */
      sts = session->SetFrameAllocator(allocator);
      IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

      externalAlloc = true;
#endif
#ifdef LIBVA_SUPPORT
      sts = createHWDevice();
      IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
      /* It's possible to skip failed result here and switch to SW implementation,
      but we don't process this way */
      mfxHDL hdl = nullptr;
      sts = hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl);
      IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
      // provide device manager to MediaSDK
      sts = session->SetHandle(MFX_HANDLE_VA_DISPLAY, hdl);
      IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
      // create VAAPI allocator
      allocator = new vaapiFrameAllocator;
      MSDK_CHECK_POINTER(allocator, MFX_ERR_MEMORY_ALLOC);

      vaapiAllocatorParams* p_vaapiAllocParams = new vaapiAllocatorParams;
      MSDK_CHECK_POINTER(p_vaapiAllocParams, MFX_ERR_MEMORY_ALLOC);

      p_vaapiAllocParams->m_dpy = (VADisplay)hdl;
      allocatorParams = p_vaapiAllocParams;

      /* In case of video memory we must provide MediaSDK with external allocator
      thus we demonstrate "external allocator" usage model.
      Call SetAllocator to pass allocator to mediasdk */
      sts = session->SetFrameAllocator(allocator);
      IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

      externalAlloc = true;
#endif
    } else {
#ifdef LIBVA_SUPPORT
      // in case of system memory allocator we also have to pass MFX_HANDLE_VA_DISPLAY to HW library
      mfxIMPL impl;
      session->QueryIMPL(&impl);

      if (MFX_IMPL_HARDWARE == MFX_IMPL_BASETYPE(impl)) {
        sts = createHWDevice();
        IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

        mfxHDL hdl = NULL;
        sts = hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, &hdl);
        IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
        // provide device manager to MediaSDK
        sts = session->SetHandle(MFX_HANDLE_VA_DISPLAY, hdl);
        IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
      }
#endif

      // create system memory allocator
      allocator = new SysMemFrameAllocator;
      MSDK_CHECK_POINTER(allocator, MFX_ERR_MEMORY_ALLOC);
      allocatorParams = nullptr;  // nullptr, so that the thing described below happens

      /*  In case of system memory we demonstrate "no external allocator" usage model.
       *  We don't call SetAllocator, Media SDK uses internal allocator.
       *  We use system memory allocator simply as a memory manager for application
       */
    }

    // initialize memory allocator
    sts = allocator->Init(allocatorParams);
    IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    return MFX_ERR_NONE;
  }

  void deleteAllocator() {
    delete allocator;
    delete allocatorParams;
    deleteHWDevice();
  }

  mfxStatus allocFrames() {
    memset(&req, 0, sizeof(req));
    memset(&resp, 0, sizeof(resp));

    mfxU16 nSurfNum = 0;

    // Calculate the number of surfaces for components.
    // QueryIOSurf functions tell how many surfaces are required to produce at least 1 output.
    // To achieve better performance we provide extra surfaces.
    // 1 extra surface at input allows to get 1 extra output.
    mfxStatus sts = decoder->QueryIOSurf(&params, &req);
    if (sts != MFX_ERR_NONE) {
      Logger::error(QSVtag) << "Unable to query IO surface with error " << sts << std::endl;
    }

    if (req.NumFrameSuggested < params.AsyncDepth) {
      return MFX_ERR_MEMORY_ALLOC;
    }

    /* move the CIRCULAR_BUFFER_LEN here to remove a memcopy for rtmp decoder synchronization */
    nSurfNum = req.NumFrameSuggested + CIRCULAR_BUFFER_LEN;

    // prepare allocation requests
    req.NumFrameSuggested = req.NumFrameMin = nSurfNum;
#if defined(WIN32) || defined(WIN64)
    memcpy_s(&(req.Info), sizeof(req.Info), &(params.mfx.FrameInfo), sizeof(mfxFrameInfo));
#else
    memcpy(&(req.Info), &(params.mfx.FrameInfo), sizeof(mfxFrameInfo));
#endif

    // alloc frames for decoder
    sts = allocator->Alloc(allocator->pthis, &req, &resp);
    if (sts != MFX_ERR_NONE) {
      Logger::error(QSVtag) << "Unable to alloc surface with error " << sts << std::endl;
    }

    // prepare mfxFrameSurface1 array for decoder
    sts = AllocBuffers(nSurfNum);
    IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    for (int i = 0; i < nSurfNum; i++) {
      // initating each frame:
#if defined(WIN32) || defined(WIN64)
      memcpy_s(&(m_pSurfaces[i].frame.Info), sizeof(m_pSurfaces[i].frame.Info), &(req.Info), sizeof(mfxFrameInfo));
#else
      memcpy(&(m_pSurfaces[i].frame.Info), &(req.Info), sizeof(mfxFrameInfo));
#endif

      if (externalAlloc) {
        m_pSurfaces[i].frame.Data.MemId = resp.mids[i];
      } else {
        sts = allocator->Lock(allocator->pthis, resp.mids[i], &(m_pSurfaces[i].frame.Data));
        IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
      }
    }
    return MFX_ERR_NONE;
  }

  void deleteFrames() {
    // delete surfaces array
    FreeBuffers();

    if (m_pCurrentFreeOutputSurface) {
      free(m_pCurrentFreeOutputSurface);
    }

    // delete frames
    if (allocator) {
      mfxFrameAllocResponse resp;
      allocator->Free(allocator->pthis, &resp);
    }
  }

  // --------------------------------

  MFXVideoSession* session = nullptr;
  MFXVideoDECODE* decoder = nullptr;
  mfxVideoParam params;

  CHWDevice* hwdev = nullptr;

  MFXFrameAllocator* allocator = nullptr;
  mfxAllocatorParams* allocatorParams = nullptr;
  MemType memType = MemType::SYSTEM_MEMORY;
  bool externalAlloc = false;  // use memory allocator as external for Media SDK

  mfxFrameAllocRequest req;
  mfxFrameAllocResponse resp;

  msdkFrameSurface* m_pCurrentFreeSurface = nullptr;         // surface detached from free surfaces array
  msdkOutputSurface* m_pCurrentFreeOutputSurface = nullptr;  // surface detached from free output surfaces array

  mfxBitstream bitS;

  FrameRate framerate;

  bool sequenceHeader = false, codecConfig = false;

  mtime_t lastTimestamp;
  mtime_t inputTimestamp;
  mtime_t latency;
};

VideoDecoder* createQSVDecoder(int width, int height, FrameRate framerate) {
  // XXX TODO FIXME map mem_type to our enum
  MemType memtype = SYSTEM_MEMORY;

  QSVDecoder* decoder = new QSVDecoder();
  if (decoder->init(width, height, framerate, memtype) == MFX_ERR_NONE) {
    return decoder;
  } else {
    Logger::error(QSVtag) << "Cannot instantiate the decoder" << std::endl;
    delete decoder;
    return nullptr;
  }
}

/** Constants **/
const std::chrono::milliseconds QSVDecoder::GET_SURFACE_TIMEOUT = std::chrono::milliseconds(10);

}  // namespace Input
}  // namespace VideoStitch
