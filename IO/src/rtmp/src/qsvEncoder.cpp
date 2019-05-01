// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include "libvideostitch/logging.hpp"

#include "librtmpIncludes.hpp"
#include "qsvEncoder.hpp"
#include "ptvMacro.hpp"

#define IMSDK_CHECK_RESULT(P, X, ERR)                                                                           \
  {                                                                                                            \
    if ((X) > (P)) {                                                                                           \
      std::stringstream msg;                                                                                   \
      msg << "[RTMP] QSV Encoder " << __FUNCTION__ << " line " << __LINE__ << " return with error code " << P; \
      Status(VideoStitch::Origin::Output, VideoStitch::ErrType::RuntimeError, msg.str());                      \
      return ERR;                                                                                              \
    }                                                                                                          \
  }

const static mfxU8 start_seq[] = {0, 0, 1};

namespace VideoStitch {
namespace Output {

const AVal QSVEncoder::av_videocodecid = mAVC("videocodecid");
const AVal QSVEncoder::av_videodatarate = mAVC("videodatarate");
const AVal QSVEncoder::av_framerate = mAVC("framerate");

const std::unordered_map<std::string, int> QSVEncoder::profileMap = {
    {"baseline", MFX_PROFILE_AVC_BASELINE},
    {"main", MFX_PROFILE_AVC_MAIN},
    {"extended", MFX_PROFILE_AVC_EXTENDED},
    {"high", MFX_PROFILE_AVC_HIGH},
    {"constrained_baseline", MFX_PROFILE_AVC_CONSTRAINED_BASELINE},
    {"constrained_high", MFX_PROFILE_AVC_CONSTRAINED_HIGH},
    {"progressive_high", MFX_PROFILE_AVC_PROGRESSIVE_HIGH}};

const std::unordered_set<mfxU16> QSVEncoder::levelSet = {
    MFX_LEVEL_AVC_1,  MFX_LEVEL_AVC_11, MFX_LEVEL_AVC_12, MFX_LEVEL_AVC_13, MFX_LEVEL_AVC_2, MFX_LEVEL_AVC_21,
    MFX_LEVEL_AVC_22, MFX_LEVEL_AVC_3,  MFX_LEVEL_AVC_31, MFX_LEVEL_AVC_32, MFX_LEVEL_AVC_4, MFX_LEVEL_AVC_41,
    MFX_LEVEL_AVC_42, MFX_LEVEL_AVC_5,  MFX_LEVEL_AVC_51, MFX_LEVEL_AVC_52};

const std::unordered_map<std::string, int> QSVEncoder::rcmodeMap = {
    {"cqp", MFX_RATECONTROL_CBR},       {"vbr", MFX_RATECONTROL_VBR},       {"cqp", MFX_RATECONTROL_CQP},
    {"avbr", MFX_RATECONTROL_AVBR},     {"la", MFX_RATECONTROL_LA},         {"icq", MFX_RATECONTROL_ICQ},
    {"vcm", MFX_RATECONTROL_VCM},       {"la_icq", MFX_RATECONTROL_LA_ICQ}, {"la_ext", MFX_RATECONTROL_LA_EXT},
    {"la_hrd", MFX_RATECONTROL_LA_HRD}, {"qvbr", MFX_RATECONTROL_QVBR}};

const std::unordered_map<std::string, MemType> QSVEncoder::memMap = {{"system", SYSTEM_MEMORY},
#if MFX_D3D11_SUPPORT
                                                                     {"d3d11", D3D11_MEMORY},
#endif
                                                                     {"d3d", D3D9_MEMORY}};

mfxStatus QSVEncoder::initSession(MFXVideoSession* session, const MemType memType) {
  // we set version to 1.0 and later we will query actual version of the library which will got leaded
  mfxVersion min_version;
  min_version.Major = 1;
  min_version.Minor = 0;
  // try searching on all display adapters
  mfxIMPL impl = MFX_IMPL_HARDWARE_ANY;
  // if d3d11 surfaces are used ask the library to run acceleration through D3D11
  // feature may be unsupported due to OS or MSDK API version
  if (D3D11_MEMORY == memType) impl |= MFX_IMPL_VIA_D3D11;
#if D3D_SURFACES_SUPPORT
  else if (D3D9_MEMORY == memType)
    impl |= MFX_IMPL_VIA_D3D9;
#elif LIBVA_SUPPORT
  else if (D3D9_MEMORY == memType)
    impl |= MFX_IMPL_VIA_VAAPI;
#endif
  mfxStatus sts = session->Init(impl, &min_version);
  // MSDK API version may not support multiple adapters - then try initialize on the default
  if (sts != MFX_ERR_NONE) {
    sts = session->Init((impl & (!MFX_IMPL_HARDWARE_ANY)) | MFX_IMPL_HARDWARE, nullptr);
  }
  return sts;
}

mfxStatus QSVEncoder::init(int width, int height, FrameRate framerate, mfxU16 profile, mfxU16 level, int bitRate,
                           mfxU16 targetUsage, mfxU16 rateControlMethod, mfxU16 numSlice, int gop, MemType memType) {
  this->bitRate = bitRate;
  this->framerate = framerate;
  this->memType = memType;

  session = new MFXVideoSession;
  mfxStatus sts = initSession(session, memType);
  if (sts != MFX_ERR_NONE) {
    return sts;
  }

  mfxVersion ver;
  mfxIMPL impl;
  session->QueryIMPL(&impl);
  session->QueryVersion(&ver);
  std::string sImpl =
      (MFX_IMPL_VIA_D3D11 == MFX_IMPL_VIA_MASK(impl)) ? "hw_d3d11" : (MFX_IMPL_HARDWARE & impl) ? "hw" : "sw";
  Logger::get(Logger::Info) << "[RTMP] Media SDK impl     " << sImpl << std::endl;
  Logger::get(Logger::Info) << "[RTMP] Media SDK version  " << ver.Major << "." << ver.Minor << std::endl;

  switch (MFX_IMPL_VIA_MASK(impl)) {
    case MFX_IMPL_VIA_D3D11:
      this->memType = D3D11_MEMORY;
      break;
    case MFX_IMPL_VIA_D3D9:
    case MFX_IMPL_VIA_VAAPI:
      this->memType = D3D9_MEMORY;
      break;
    default:
      this->memType = SYSTEM_MEMORY;
      break;
  }

  if (this->memType != memType) {
    Logger::warning("RTMP") << "Supported Memory type " << this->memType << " is not the requested one " << memType
                            << std::endl;
  }

  // create and init frame allocator
  sts = createAllocator();
  if (sts != MFX_ERR_NONE) {
    std::stringstream msg;
    msg << "[RTMP] QSV Encoder : unable to create surface allocator with error " << sts;
    Status(VideoStitch::Origin::Output, VideoStitch::ErrType::RuntimeError, msg.str());
    return sts;
  }
  // parameters for the encoding
  initEncoderParams(width, height, framerate, profile, level, bitRate, targetUsage, rateControlMethod, numSlice, gop);
  encoder = new MFXVideoENCODE(*session);

  sts = allocFrames();
  IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

  sts = encoder->Init(&params);
  if (sts == MFX_WRN_INCOMPATIBLE_VIDEO_PARAM) {
    mfxVideoParam paramsOut = params;
    encoder->Query(&params, &paramsOut);
    if (params.mfx.TargetKbps != paramsOut.mfx.TargetKbps) {
      Logger::get(Logger::Warning) << "[RTMP] Intel Quick Sync cannot use a target bitrate of " << params.mfx.TargetKbps
                                   << " Kbps for this resolution, overriding "
                                   << "to " << paramsOut.mfx.TargetKbps << " Kbps" << std::endl;
    }
  }

  // Retrieve video parameters selected by encoder.
  // BufferSizeInKB parameter is required to set bit stream buffer size
  mfxVideoParam videoParams;
  memset(&videoParams, 0, sizeof(videoParams));

  // retrieve SPS/PPS selected by encoder to store them in headPkt
  // It will be sent in the RTMP configuration packet
  headPkt.resize(256);
  mfxExtCodingOptionSPSPPS extSPSPPS;
  memset(&extSPSPPS, 0, sizeof(extSPSPPS));
  extSPSPPS.Header.BufferId = MFX_EXTBUFF_CODING_OPTION_SPSPPS;
  extSPSPPS.Header.BufferSz = sizeof(mfxExtCodingOptionSPSPPS);
  extSPSPPS.PPSBufSize = (mfxU16)headPkt.size() / 2;
  extSPSPPS.SPSBufSize = (mfxU16)headPkt.size() / 2;
  extSPSPPS.PPSBuffer = headPkt.data() + extSPSPPS.SPSBufSize;
  extSPSPPS.SPSBuffer = headPkt.data();
  mfxExtBuffer* encExtParams[1];
  encExtParams[0] = (mfxExtBuffer*)&extSPSPPS;
  videoParams.ExtParam = &encExtParams[0];
  videoParams.NumExtParam = 1;

  sts = encoder->GetVideoParam(&videoParams);
  IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
  if (sts == MFX_ERR_NONE) {
    memcpy(extSPSPPS.SPSBuffer + extSPSPPS.SPSBufSize, extSPSPPS.PPSBuffer, extSPSPPS.PPSBufSize);
    headPkt.resize(extSPSPPS.SPSBufSize + extSPSPPS.PPSBufSize);
  }

  bits.MaxLength = videoParams.mfx.BufferSizeInKB * 1024;
  bits.Data = new mfxU8[bits.MaxLength];
  bits.DataOffset = 0;
  bits.DataLength = 0;

  return MFX_ERR_NONE;
}

bool QSVEncoder::encode(const Frame& videoFrame, std::vector<VideoStitch::IO::DataPacket>& packets) {
  mfxSyncPoint syncp = nullptr;

  // find an unlocked surface
  mfxU16 encSurfIdx = getFreeSurface(encSurfaces, resp.NumFrameActual);
  mfxFrameSurface1* surf = &encSurfaces[encSurfIdx];

  mfxStatus sts;
  if (externalAlloc) {
    // if we share allocator with Media SDK we need to call Lock to access surface data
    sts = allocator->Lock(allocator->pthis, surf->Data.MemId, &(surf->Data));
    IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, false);
  }
  mfxFrameInfo& pInfo = surf->Info;
  mfxFrameData& pData = surf->Data;

  // fill content for encoding
  // the unit used by the MFX framework is 90 Khz
  surf->Data.TimeStamp = mfxU64(std::round(9 * videoFrame.pts / 100.0));
  const char* in_ptr = (const char*)videoFrame.planes[0];
  mfxU16 w, h;
  mfxU16 pitch = pData.Pitch;
  mfxU8* ptr;
  ptr = pData.Y + pInfo.CropX + pInfo.CropY * pData.Pitch;
  if (pInfo.CropH > 0 && pInfo.CropW > 0) {
    w = pInfo.CropW;
    h = pInfo.CropH;
  } else {
    w = pInfo.Width;
    h = pInfo.Height;
  }
  // luminance plane
  for (auto i = 0; i < h; i++) {
    memcpy(ptr + i * pitch, in_ptr, w);
    in_ptr += w;
  }
  // chroma planes
  h /= 2;
  ptr = pData.UV + pInfo.CropX + (pInfo.CropY / 2) * pitch;
  in_ptr = (const char*)videoFrame.planes[1];
  for (auto i = 0; i < h; i++) {
    memcpy(ptr + i * pitch, in_ptr, w);
    in_ptr += w;
  }
  if (externalAlloc) {
    sts = allocator->Unlock(allocator->pthis, surf->Data.MemId, &(surf->Data));
    IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, false);
  }

  sts = encoder->EncodeFrameAsync(nullptr, surf, &bits, &syncp);
  if (sts == MFX_ERR_NONE) {
    sts = session->SyncOperation(syncp, INFINITE);
    if (sts != MFX_ERR_NONE) {
      return false;
    }

    // get those packets!
    std::vector<x264_nal_t> nalOut;
    x264_nal_t nal;
    switch ((bits.FrameType & (MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_REF | (MFX_FRAMETYPE_S - 1)))) {
      case MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_REF | MFX_FRAMETYPE_I:
        nal.i_type = NAL_SLICE_IDR;
        nal.i_ref_idc = NAL_PRIORITY_HIGHEST;
        break;
      case MFX_FRAMETYPE_REF | MFX_FRAMETYPE_I:
      case MFX_FRAMETYPE_REF | MFX_FRAMETYPE_P:
        nal.i_type = NAL_SLICE;
        nal.i_ref_idc = NAL_PRIORITY_HIGH;
        break;
      case MFX_FRAMETYPE_REF | MFX_FRAMETYPE_B:
        nal.i_type = NAL_SLICE;
        nal.i_ref_idc = NAL_PRIORITY_LOW;
        break;
      case MFX_FRAMETYPE_B:
        nal.i_type = NAL_SLICE;
        nal.i_ref_idc = NAL_PRIORITY_DISPOSABLE;
        break;
      default:
        Logger::warning("RTMP") << "Unhandled frametype " << bits.FrameType << std::endl;
        break;
    }
    nal.p_payload = (uint8_t*)bits.Data + bits.DataOffset;
    nal.i_payload = int(bits.DataLength);
    nalOut.push_back(nal);
    bits.DataLength = 0;

    VideoEncoder::createDataPacket(nalOut, packets, mtime_t(std::round(bits.TimeStamp / 90.0)),
                                   mtime_t(std::round(bits.DecodeTimeStamp / 90.0)));

    return true;
  } else if (sts == MFX_ERR_MORE_DATA) {
    // feed more frames to the encoder!
    return true;
  } else {
    return false;
  }
}

char* QSVEncoder::metadata(char* enc, char* pend) {
  enc = AMF_EncodeNamedNumber(enc, pend, &av_videocodecid, 7.);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_videodatarate, double(bitRate));
  enc = AMF_EncodeNamedNumber(enc, pend, &av_framerate, double(framerate.num) / double(framerate.den));
  return enc;
}

mfxU16 QSVEncoder::getFreeSurface(mfxFrameSurface1* surfacesPool, mfxU16 poolSize) {
  mfxU32 sleepInterval = 10;  // milliseconds
  mfxU16 idx = 0xFFFF;

  // wait if there's no free surface
  for (mfxU32 i = 0; i < 0xFFFF; i += sleepInterval) {
    idx = getFreeSurfaceIndex(surfacesPool, poolSize);

    if (0xFFFF != idx) {
      break;
    } else {
#if defined(_WIN32) || defined(_WIN64)
      Sleep(sleepInterval);
#else
      usleep(1000 * sleepInterval);
#endif
    }
  }
  return idx;
}

mfxU16 QSVEncoder::getFreeSurfaceIndex(mfxFrameSurface1* surfacesPool, mfxU16 poolSize) {
  if (surfacesPool) {
    for (mfxU16 i = 0; i < poolSize; i++) {
      if (0 == surfacesPool[i].Data.Locked) {
        return i;
      }
    }
  }
  return 0xFFFF;
}

void QSVEncoder::initEncoderParams(int width, int height, FrameRate framerate, mfxU16 profile, mfxU16 level,
                                   int bitRate, mfxU16 targetUsage, mfxU16 rateControlMethod, mfxU16 numSlice,
                                   int gop) {
  memset(&params, 0, sizeof(params));

  params.mfx.CodecId = MFX_CODEC_AVC;
  params.mfx.CodecProfile = profile;
  params.mfx.CodecLevel = level;
  params.mfx.TargetUsage = targetUsage;  // trade-off between quality and speed, from 1 (quality) to 7 (fastest)
  params.mfx.TargetKbps = bitRate;       // in Kbps
  params.mfx.RateControlMethod = rateControlMethod;
  params.mfx.NumSlice = numSlice;
  params.mfx.FrameInfo.FrameRateExtN = framerate.num;
  params.mfx.FrameInfo.FrameRateExtD = framerate.den;
  params.mfx.EncodedOrder = 0;  // binary flag, 0 signals encoder to take frames in display order
  if (gop >= 0) {
    params.mfx.GopPicSize = gop;
  }
  params.mfx.IdrInterval = 0;  // closed GOP, every I-Frame is an IDR-Frame => better to catch live stream

  // specify memory type
  if (memType == D3D9_MEMORY || memType == D3D11_MEMORY) {
    params.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY;
  } else {
    params.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;
  }

  // frame info parameters
  params.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
  params.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
  params.mfx.FrameInfo.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;

  // set frame size and crops
  // width must be a multiple of 16
  // height must be a multiple of 16 in case of frame picture
  params.mfx.FrameInfo.Width = MSDK_ALIGN16(width);
  params.mfx.FrameInfo.Height = MSDK_ALIGN16(height);

  params.mfx.FrameInfo.CropX = 0;
  params.mfx.FrameInfo.CropY = 0;
  params.mfx.FrameInfo.CropW = width;
  params.mfx.FrameInfo.CropH = height;
}

mfxStatus QSVEncoder::createHWDevice() {
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
      Logger::get(Logger::Error) << "[RTMP] QSV Encoder : unable to init mfx dispatcher lib with error " << sts
                                 << std::endl;
    }

    sts = MFXQueryIMPL(auxSession, &impl);
    if (sts != MFX_ERR_NONE) {
      Logger::get(Logger::Error) << "[RTMP] QSV Encoder : unable to get mfx dispatcher implementation with error "
                                 << sts << std::endl;
    }
    sts = MFXClose(auxSession);
    if (sts != MFX_ERR_NONE) {
      Logger::get(Logger::Error) << "[RTMP] QSV Encoder : unable to init encoder with error " << sts << std::endl;
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
  if (D3D11_MEMORY == memType)
    hwdev = new CD3D11Device();
  else
#endif  // #if MFX_D3D11_SUPPORT
    hwdev = new CD3D9Device();

  if (hwdev == nullptr) return MFX_ERR_MEMORY_ALLOC;

  sts = hwdev->Init(
      nullptr, 0,
      adapterNum);  // XXX TODO FIXME https://software.intel.com/en-us/forums/intel-media-sdk/topic/599935#node-599935

#elif LIBVA_SUPPORT
  hwdev = CreateVAAPIDevice();
  if (hwdev == nullptr) {
    return MFX_ERR_MEMORY_ALLOC;
  }
  sts = hwdev->Init(nullptr, 0, adapterNum);
#endif
  return sts;
}

mfxStatus QSVEncoder::resetDevice() {
  if (D3D9_MEMORY == memType || D3D11_MEMORY == memType) {
    return hwdev->Reset();
  }
  return MFX_ERR_NONE;
}

mfxStatus QSVEncoder::createAllocator() {
  mfxStatus sts = MFX_ERR_NONE;

  if (D3D9_MEMORY == memType || D3D11_MEMORY == memType) {
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

    /* In case of system memory we demonstrate "no external allocator" usage model.
    We don't call SetAllocator, Media SDK uses internal allocator.
    We use system memory allocator simply as a memory manager for application*/
  }

  // initialize memory allocator
  sts = allocator->Init(allocatorParams);
  IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

  return MFX_ERR_NONE;
}

void QSVEncoder::deleteAllocator() {
  delete allocator;
  delete allocatorParams;
  deleteHWDevice();
}

mfxStatus QSVEncoder::allocFrames() {
  memset(&req, 0, sizeof(req));
  memset(&resp, 0, sizeof(resp));

  mfxU16 nEncSurfNum = 0;

  // Calculate the number of surfaces for components.
  // QueryIOSurf functions tell how many surfaces are required to produce at least 1 output.
  // To achieve better performance we provide extra surfaces.
  // 1 extra surface at input allows to get 1 extra output.
  mfxStatus sts = encoder->QueryIOSurf(&params, &req);
  IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

  if (req.NumFrameSuggested < params.AsyncDepth) return MFX_ERR_MEMORY_ALLOC;

  // The number of surfaces shared by vpp output and encode input.
  nEncSurfNum = req.NumFrameSuggested;

  // prepare allocation requests
  req.NumFrameSuggested = req.NumFrameMin = nEncSurfNum;
#if defined(WIN32) || defined(WIN64)
  memcpy_s(&(req.Info), sizeof(req.Info), &(params.mfx.FrameInfo), sizeof(mfxFrameInfo));
#else
  memcpy(&(req.Info), &(params.mfx.FrameInfo), sizeof(mfxFrameInfo));
#endif

  // alloc frames for encoder
  sts = allocator->Alloc(allocator->pthis, &req, &resp);
  IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

  // prepare mfxFrameSurface1 array for encoder
  encSurfaces = new mfxFrameSurface1[resp.NumFrameActual];
  MSDK_CHECK_POINTER(encSurfaces, MFX_ERR_MEMORY_ALLOC);

  for (int i = 0; i < resp.NumFrameActual; i++) {
    memset(&(encSurfaces[i]), 0, sizeof(mfxFrameSurface1));
#if defined(WIN32) || defined(WIN64)
    memcpy_s(&(encSurfaces[i].Info), sizeof(encSurfaces[i].Info), &(params.mfx.FrameInfo), sizeof(mfxFrameInfo));
#else
    memcpy(&(encSurfaces[i].Info), &(params.mfx.FrameInfo), sizeof(mfxFrameInfo));
#endif

    if (externalAlloc) {
      encSurfaces[i].Data.MemId = resp.mids[i];
    } else {
      // get YUV pointers
      sts = allocator->Lock(allocator->pthis, resp.mids[i], &(encSurfaces[i].Data));
      IMSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
    }
  }
  return MFX_ERR_NONE;
}

void QSVEncoder::deleteFrames() {
  // delete surfaces array
  delete[] encSurfaces;

  // delete frames
  if (allocator) {
    mfxFrameAllocResponse resp;
    allocator->Free(allocator->pthis, &resp);
  }
}

mfxU16 QSVEncoder::profileParser(std::string profile) {
  if (profileMap.find(profile) != profileMap.end()) {
    return profileMap.at(profile);
  }
  return MFX_PROFILE_AVC_BASELINE;
}

mfxU16 QSVEncoder::levelParser(std::string level) {
  std::string::size_type index;
  int firstNumber = std::stoi(level, &index);
  int secondNumber = 0;
  if (index < level.size()) {
    secondNumber = std::stoi(level.substr(index + 1));
  }
  mfxU16 levelNB = firstNumber * 10 + secondNumber;

  if (levelSet.find(levelNB) != levelSet.end()) {
    return levelNB;
  }
  return MFX_LEVEL_AVC_42;
}

mfxU16 QSVEncoder::rcmodeParser(std::string rc_mode) {
  std::transform(rc_mode.begin(), rc_mode.end(), rc_mode.begin(), ::tolower);
  if (rcmodeMap.find(rc_mode) != rcmodeMap.end()) {
    return rcmodeMap.at(rc_mode);
  }
  return MFX_RATECONTROL_CBR;
}

MemType QSVEncoder::memTypeParser(std::string memtype) {
  std::transform(memtype.begin(), memtype.end(), memtype.begin(), ::tolower);
  if (memMap.find(memtype) != memMap.end()) {
    return memMap.at(memtype);
  }
#if MFX_D3D11_SUPPORT
  return D3D11_MEMORY;
#else
  return D3D9_MEMORY;
#endif
}

void QSVEncoder::supportedEncoders(std::vector<std::string>& codecs) {
  MFXVideoSession* session = new MFXVideoSession;
#if MFX_D3D11_SUPPORT
  MemType memType = D3D11_MEMORY;
#else
  MemType memType = D3D9_MEMORY;
#endif

  mfxStatus sts = initSession(session, memType);
  if (sts == MFX_ERR_NONE) {
    codecs.push_back("qsv");
  } else {
    mfxVersion ver;
    mfxIMPL impl;
    session->QueryIMPL(&impl);
    session->QueryVersion(&ver);
    std::string sImpl =
        (MFX_IMPL_VIA_D3D11 == MFX_IMPL_VIA_MASK(impl)) ? "hw_d3d11" : (MFX_IMPL_HARDWARE & impl) ? "hw" : "sw";
    Logger::info("RTMP") << "Media SDK : " << sImpl << " implementation version " << ver.Major << "." << ver.Minor
                         << std::endl;
    Logger::info("RTMP") << "Media SDK : QSV Encoder Hardware acceleration not available" << std::endl;
  }
  delete session;
}

Potential<VideoEncoder> QSVEncoder::createQSVEncoder(const Ptv::Value& config, int width, int height,
                                                     FrameRate framerate) {
  INT(config, bitrate, 4000);    // kbps
  INT(config, target_usage, 4);  // trade-off between quality and speed, from 1 (quality) to 7 (fastest)
  INT(config, num_slice, 0);     // number of slices in each video frame.
  // If num_slice equals zero, the encoder may choose any slice partitioning allowed by the codec standard.
  STRING(config, bitrate_mode, "cbr");
  STRING(config, mem_type, "d3d11");
  STRING(config, profile, "baseline");
  STRING(config, level, "3.1");
  INT(config, gop, 250);

  MemType memtype = memTypeParser(mem_type);
  int rate_control_method = rcmodeParser(bitrate_mode);

  auto encoder = new QSVEncoder();
  if (encoder->init(width, height, framerate, QSVEncoder::profileParser(profile), QSVEncoder::levelParser(level),
                    bitrate, target_usage, rate_control_method, num_slice, gop, memtype) == MFX_ERR_NONE) {
    return Potential<VideoEncoder>(encoder);
  } else {
    delete encoder;
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration,
                                   "[RTMP] could not instantiate the Quick Sync encoder");
  }
}

}  // namespace Output
}  // namespace VideoStitch
