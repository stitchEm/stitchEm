// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <algorithm>
#include "nvenc.hpp"
#include "ptvMacro.hpp"

#include <sstream>
#include "libvideostitch/logging.hpp"

#include <cuda.h>

#define FABS(a) ((a) >= 0 ? (a) : -(a))

#define CHECK(fn)                                                                    \
  if (nvStatus != NV_ENC_SUCCESS) {                                                  \
    Logger::warning(__NVENC__) << #fn " failed with code " << nvStatus << std::endl; \
  }
#define CHECK_RETURN(fn)                                                             \
  if (nvStatus != NV_ENC_SUCCESS) {                                                  \
    Logger::warning(__NVENC__) << #fn " failed with code " << nvStatus << std::endl; \
    return nvStatus;                                                                 \
  }

inline void nvCloseFile(HANDLE hFileHandle) {
  if (hFileHandle) {
#if defined(_WIN32)
    CloseHandle(hFileHandle);
#else
    fclose((FILE*)hFileHandle);
#endif
  }
}

const std::string __NVENC__("NVENC");

namespace VideoStitch {
namespace Output {

static inline void resetCudaError() { cudaGetLastError(); }

NvEncoder::NvEncoder(EncodeConfig& config) : encodeConfig(config) {
  m_hEncoder = NULL;
  m_bEncoderInitialized = false;
  m_pEncodeAPI = NULL;
  m_hinstLib = NULL;
  m_fOutput = config.fOutput;
  first_dts = 0;

  bitrateMax = config.bitrate;

  cudaError_t r = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::error(__NVENC__) << "cudaStreamCreateWithFlags has returned CUDA error " << r << std::endl;
  }
  memset(&m_stCreateEncodeParams, 0, sizeof(m_stCreateEncodeParams));
  SET_VER(m_stCreateEncodeParams, NV_ENC_INITIALIZE_PARAMS);

  memset(&m_stEncodeConfig, 0, sizeof(m_stEncodeConfig));
  SET_VER(m_stEncodeConfig, NV_ENC_CONFIG);

  m_uEncodeBufferCount = 0;
  memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));

  memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));
}

NvEncoder::~NvEncoder() {
  NVENCSTATUS nvStatus;
  cudaError_t r = cudaStreamSynchronize(stream);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::warning(__NVENC__) << "cudaStreamSynchronize has returned CUDA error " << r << std::endl;
  }

  nvStatus = FlushEncoder();
  CHECK(FlushEncoder)

  nvStatus = ReleaseIOBuffers();
  CHECK(ReleaseIOBuffers)

  nvStatus = NvEncDestroyEncoder();
  CHECK(NvEncDestroyEncoder)

  // clean up encode API resources here
  if (m_pEncodeAPI) {
    delete m_pEncodeAPI;
    m_pEncodeAPI = NULL;
  }

  if (m_hinstLib) {
#if defined(_WIN32)
    FreeLibrary(m_hinstLib);
#else
    dlclose(m_hinstLib);
#endif

    m_hinstLib = NULL;
  }

  r = cudaStreamDestroy(stream);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::warning(__NVENC__) << "cudaStreamDestroy failed with code " << r << std::endl;
  }

  if (m_fOutput != NULL) {
    fclose(m_fOutput);
  }
}

void NvEncoder::supportedEncoders(std::vector<std::string>& codecs) {
  GUID* GUIDs = nullptr;
  EncodeConfig encodeConfig = {};
  auto encoder = new NvEncoder(encodeConfig);
  NVENCSTATUS nvStatus = encoder->InitCuda(encodeConfig.deviceID);
  if (nvStatus != NV_ENC_SUCCESS) {
    Logger::warning(__NVENC__) << "InitCuda failed with code " << nvStatus << std::endl;
    goto end;
  }

#if defined __linux__
  // NVENC triggers a leak in linux_lsan, probably in libnvidia-encode, impossible to debug
  codecs.push_back("h264_nvenc");
  codecs.push_back("hevc_nvenc");
#else
  nvStatus = encoder->InitializeAPI();
  if (nvStatus != NV_ENC_SUCCESS) {
    Logger::warning(__NVENC__) << "Initialize API failed with code " << nvStatus << std::endl;
    goto end;
  }

  CUcontext ctx;
  cuCtxGetCurrent(&ctx);
  nvStatus = encoder->NvEncOpenEncodeSessionEx((void*)ctx, NV_ENC_DEVICE_TYPE_CUDA);
  if (nvStatus != NV_ENC_SUCCESS) {
    Logger::warning(__NVENC__) << "Open session failed with code " << nvStatus << std::endl;
    goto end;
  }

  uint32_t encodeGUIDTotal;
  nvStatus = encoder->NvEncGetEncodeGUIDCount(&encodeGUIDTotal);
  if (nvStatus != NV_ENC_SUCCESS) {
    Logger::warning(__NVENC__) << "NvEncGetEncodeGUIDCount failed with code " << nvStatus << std::endl;
    goto end;
  }

  uint32_t encodeGUIDCount;
  GUIDs = (GUID*)malloc(encodeGUIDTotal * sizeof(GUID));
  if (GUIDs == nullptr) {
    Logger::warning(__NVENC__) << "Could not allocate GUIDs" << std::endl;
    goto end;
  }
  nvStatus = encoder->NvEncGetEncodeGUIDs(GUIDs, encodeGUIDTotal, &encodeGUIDCount);
  if (nvStatus != NV_ENC_SUCCESS) {
    Logger::warning(__NVENC__) << "NvEncGetEncodeGUIDs failed with code " << nvStatus << std::endl;
    goto end;
  }

  for (int i = 0; i < (int)encodeGUIDCount; i++) {
    if (GUIDs[i] == NV_ENC_CODEC_H264_GUID) {
      codecs.push_back("h264_nvenc");
    } else if (GUIDs[i] == NV_ENC_CODEC_HEVC_GUID) {
      codecs.push_back("hevc_nvenc");
    }
  }
#endif

end:
  free((void*)GUIDs);
  delete encoder;
}

Potential<VideoEncoder> NvEncoder::createNvEncoder(const Ptv::Value& config, int width, int height,
                                                   FrameRate framerate) {
  std::string bitrate_mode = "";
  std::string filename = "";
  EncodeConfig encodeConfig = {};
  encodeConfig.endFrameIdx = INT_MAX;
  encodeConfig.bitrate = 5000;  // in Kbps
  encodeConfig.rcMode = NV_ENC_PARAMS_RC_CONSTQP;
  encodeConfig.deviceType = 2;  // CUDA interop
  encodeConfig.codec = NV_ENC_H264;
  encodeConfig.fps = framerate;
  encodeConfig.gopLength = 250;
  encodeConfig.qp = 28;
  encodeConfig.i_quant_factor = DEFAULT_I_QFACTOR;
  encodeConfig.b_quant_factor = DEFAULT_B_QFACTOR;
  encodeConfig.i_quant_offset = DEFAULT_I_QOFFSET;
  encodeConfig.b_quant_offset = DEFAULT_B_QOFFSET;
  encodeConfig.encoderProfile = "baseline";
  encodeConfig.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  encodeConfig.width = width;
  encodeConfig.height = height;
  encodeConfig.fOutput = NULL;

  if (Parse::populateInt("RTMP", config, "bitrate", encodeConfig.bitrate, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid bitrate type (int) in Kbps.");
  }
  if (Parse::populateInt("RTMP", config, "vbvMaxBitrate", encodeConfig.vbvMaxBitrate, false) ==
      Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid vbvMaxBitrate (int) in Kbps.");
  }
  if (Parse::populateInt("RTMP", config, "buffer_size", encodeConfig.vbvSize, false) ==
      Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid vbvSize (int) in Kb.");
  }
  if (Parse::populateString("RTMP", config, "bitrate_mode", bitrate_mode, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid rate control mode type (string).");
  }
  if (bitrate_mode != "") {
    if (stricmp(bitrate_mode.c_str(), "cqp") == 0) {
      encodeConfig.rcMode = NV_ENC_PARAMS_RC_CONSTQP;
    } else if (stricmp(bitrate_mode.c_str(), "cbr") == 0) {
      encodeConfig.rcMode = NV_ENC_PARAMS_RC_CBR;
    } else if (stricmp(bitrate_mode.c_str(), "vbr") == 0) {
      encodeConfig.rcMode = NV_ENC_PARAMS_RC_VBR;
    } else if (stricmp(bitrate_mode.c_str(), "vbr_minqp") == 0) {
      encodeConfig.rcMode = NV_ENC_PARAMS_RC_VBR_MINQP;
    }
  }
  if (Parse::populateInt("RTMP", config, "rcmode", encodeConfig.rcMode, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid rcmode type (int).");
  }
  if (Parse::populateInt("RTMP", config, "gop", encodeConfig.gopLength, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid goplength type (int).");
  }
  if (Parse::populateInt("RTMP", config, "b_frames", encodeConfig.numB, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid numB type (int).");
  }
  if (Parse::populateInt("RTMP", config, "qp", encodeConfig.qp, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid qp type (int).");
  }
  if (Parse::populateString("RTMP", config, "preset", encodeConfig.encoderPreset, false) ==
      Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid preset type (string). Supported presets: slow, medium, fast, hp, hq, bd, "
                                   "ll, llhp, llhq, lossless, losslesshp, default.");
  }
  if (Parse::populateString("RTMP", config, "profile", encodeConfig.encoderProfile, false) ==
      Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid profile type (string).");
  }
  std::string codec;
  if (Parse::populateString("RTMP", config, "codec", codec, false) == VideoStitch::Parse::PopulateResult_Ok) {
    Logger::info(__NVENC__) << "'codec' parameter deprecated, use 'video_codec' instead" << std::endl;
  }
  if (Parse::populateString("RTMP", config, "video_codec", codec, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid video_codec type (string). Supported codecs: h264_nvenc, hevc_nvenc");
  }

  if ((codec.find("h264") != std::string::npos) || (codec == "nvenc")) {
    encodeConfig.codec = NV_ENC_H264;
  } else if (codec.find("hevc") != std::string::npos) {
    encodeConfig.codec = NV_ENC_HEVC;
  }

  if (encodeConfig.encoderPreset != "") {
    if ((encodeConfig.encoderPreset == "hq") || (encodeConfig.encoderPreset == "slow")) {
      encodeConfig.encoderPreset = "hq";
    } else if ((encodeConfig.encoderPreset == "hp") || (encodeConfig.encoderPreset == "fast")) {
      encodeConfig.encoderPreset = "hp";
    } else if ((encodeConfig.encoderPreset == "bd") || (encodeConfig.encoderPreset == "ll") ||
               (encodeConfig.encoderPreset == "llhp") || (encodeConfig.encoderPreset == "llhq") ||
               (encodeConfig.encoderPreset == "lossless") || (encodeConfig.encoderPreset == "losslesshp")) {
    } else if ((encodeConfig.encoderPreset == "default") || (encodeConfig.encoderPreset == "medium")) {
      encodeConfig.encoderPreset = "default";
    } else {
      std::stringstream msg;
      msg << "Invalid preset : " << encodeConfig.encoderPreset
          << ". Supported presets: slow, medium, fast, hp, hq, bd, ll, llhp, llhq, lossless, losslesshp, default.";
      return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                     msg.str());
    }
  }

  if (Parse::populateString("RTMP", config, "bitstream_name", filename, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid file name type (string).");
  }
  if (filename != "") {
    encodeConfig.fOutput = fopen(filename.c_str(), "wb");
  }

  auto encoder = new NvEncoder(encodeConfig);
  Status initEnc = encoder->Initialize();
  if (initEnc.ok()) {
    return Potential<VideoEncoder>(encoder);
  } else {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "could not instantiate the NVENC encoder", initEnc);
  }
}

bool NvEncoder::setBitRate(uint32_t bitrate, uint32_t bufferSize) {
  NV_ENC_RECONFIGURE_PARAMS reconfParams;
  memset(&reconfParams, 0, sizeof(reconfParams));
  SET_VER(reconfParams, NV_ENC_RECONFIGURE_PARAMS);
  memcpy(&reconfParams.reInitEncodeParams, &m_stCreateEncodeParams, sizeof(m_stCreateEncodeParams));
  if (bitrate) {
    if (m_stEncodeConfig.rcParams.maxBitRate == m_stEncodeConfig.rcParams.averageBitRate) {
      m_stEncodeConfig.rcParams.maxBitRate = 1000 * bitrate;
    }
    m_stEncodeConfig.rcParams.averageBitRate = 1000 * bitrate;
  }
  if (bufferSize != uint32_t(-1)) {
    m_stEncodeConfig.rcParams.maxBitRate = 1000 * bitrate;
    m_stEncodeConfig.rcParams.vbvBufferSize = 1000 * bufferSize;
  }

  NVENCSTATUS nvStatus = NvEncReconfigureEncoder(&reconfParams);
  if (nvStatus != NV_ENC_SUCCESS) {
    Logger::error(__NVENC__) << "Failed to setBitRate  " << nvStatus << std::endl;
    return false;
  } else {
    Logger::info(__NVENC__) << "Set bit rate to " << bitrate << std::endl;
    return true;
  }
}

bool NvEncoder::encode(const Frame& frame, std::vector<VideoStitch::IO::DataPacket>& packets) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

  EncodeBuffer* pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
  if (!pEncodeBuffer) {
    pEncodeBuffer = m_EncodeBufferQueue.GetPending();
    nvStatus = ProcessOutput(pEncodeBuffer, packets);
    if (nvStatus != NV_ENC_SUCCESS) {
      Logger::error(__NVENC__) << "Failed to ProcessOutput  " << nvStatus << std::endl;
      return false;
    }
    // UnMap the input buffer after frame done
    if (pEncodeBuffer->stInputBfr.hInputSurface) {
      nvStatus = NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
      if (nvStatus != NV_ENC_SUCCESS) {
        Logger::error(__NVENC__) << "Failed to Unmap input buffer  " << pEncodeBuffer->stInputBfr.hInputSurface
                                 << std::endl;
        return false;
      }
      pEncodeBuffer->stInputBfr.hInputSurface = NULL;
    }
    pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
  }

  // copy luma
  cudaError_t r =
      cudaMemcpy2DAsync(pEncodeBuffer->stInputBfr.pNV12devPtr, pEncodeBuffer->stInputBfr.uNV12Stride, frame.planes[0],
                        frame.pitches[0], encodeConfig.width, encodeConfig.height, cudaMemcpyDeviceToDevice, stream);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::get(Logger::Error) << "cudaMemcpy2DAsync has returned CUDA error " << r << std::endl;
    return false;
  }
  // copy chroma
  r = cudaMemcpy2DAsync((unsigned char*)pEncodeBuffer->stInputBfr.pNV12devPtr +
                            pEncodeBuffer->stInputBfr.uNV12Stride * encodeConfig.height,
                        pEncodeBuffer->stInputBfr.uNV12Stride, frame.planes[1], frame.pitches[1], encodeConfig.width,
                        encodeConfig.height / 2, cudaMemcpyDeviceToDevice, stream);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::error(__NVENC__) << "cudaMemcpy2DAsync has returned CUDA error " << r << std::endl;
    return false;
  }
  r = cudaStreamSynchronize(stream);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::error(__NVENC__) << "cudaStreamSynchronize has returned CUDA error " << r << std::endl;
    return false;
  }

  nvStatus =
      NvEncMapInputResource(pEncodeBuffer->stInputBfr.nvRegisteredResource, &pEncodeBuffer->stInputBfr.hInputSurface);
  if (nvStatus != NV_ENC_SUCCESS) {
    Logger::error(__NVENC__) << "Failed to Map input buffer  " << pEncodeBuffer->stInputBfr.hInputSurface << std::endl;
    return false;
  }

  timestamps.push(frame.pts);

  nvStatus = NvEncEncodeFrame(pEncodeBuffer, NULL, encodeConfig.width, encodeConfig.height, frame.pts);
  if (nvStatus != NV_ENC_SUCCESS) {
    Logger::error(__NVENC__) << "Failed to encode input frame  " << pEncodeBuffer->stInputBfr.hInputSurface
                             << std::endl;
    return false;
  }

  return true;
}

const AVal NvEncoder::av_videocodecid = mAVC("videocodecid");
const AVal NvEncoder::av_videodatarate = mAVC("videodatarate");
const AVal NvEncoder::av_framerate = mAVC("framerate");

char* NvEncoder::metadata(char* enc, char* pend) {
  switch (encodeConfig.codec) {
    case NV_ENC_H264:
      enc = AMF_EncodeNamedNumber(enc, pend, &av_videocodecid, 7.);
      break;
    case NV_ENC_HEVC:
      // Intel LOL
      enc = AMF_EncodeNamedNumber(enc, pend, &av_videocodecid, 8.);
      break;
  }
  enc = AMF_EncodeNamedNumber(enc, pend, &av_videodatarate, double(encodeConfig.bitrate));
  enc = AMF_EncodeNamedNumber(enc, pend, &av_framerate, double(encodeConfig.fps.num) / (double)encodeConfig.fps.den);
  return enc;
}

// ----------- private

extern "C" {
#if defined(_WIN32)
#include "x264/x264.h"
#else
#include <unistd.h>
#include <inttypes.h>
#include <x264.h>

#define INFINITE 0xFFFFFFFF
#endif

NVENCSTATUS NvEncoder::ProcessOutput(const EncodeBuffer* pEncodeBuffer,
                                     std::vector<VideoStitch::IO::DataPacket>& packets) {
  if (pEncodeBuffer->stOutputBfr.hBitstreamBuffer == NULL && pEncodeBuffer->stOutputBfr.bEOSFlag == FALSE) {
    return NV_ENC_ERR_INVALID_PARAM;
  }

  if (pEncodeBuffer->stOutputBfr.bWaitOnEvent == TRUE) {
    if (!pEncodeBuffer->stOutputBfr.hOutputEvent) {
      return NV_ENC_ERR_INVALID_PARAM;
    }
#if defined(_WIN32)
    WaitForSingleObject(pEncodeBuffer->stOutputBfr.hOutputEvent, INFINITE);
#endif
  }

  if (pEncodeBuffer->stOutputBfr.bEOSFlag) {
    return NV_ENC_SUCCESS;
  }

  mtime_t dts = timestamps.front();
  /* when there're b frame(s), set dts offset */
  if ((m_stEncodeConfig.frameIntervalP >= 2) && (first_dts == 0)) {
    dts = (mtime_t)(dts - ((1000000.0 * m_stCreateEncodeParams.frameRateDen) / m_stCreateEncodeParams.frameRateNum));
    first_dts = 1;
  } else {
    timestamps.pop();
  }

  NV_ENC_LOCK_BITSTREAM lockBitstreamData;
  memset(&lockBitstreamData, 0, sizeof(lockBitstreamData));
  SET_VER(lockBitstreamData, NV_ENC_LOCK_BITSTREAM);
  lockBitstreamData.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
  lockBitstreamData.doNotWait = false;

  NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, &lockBitstreamData);
  CHECK(nvEncLockBitstream)
  if (nvStatus == NV_ENC_SUCCESS) {
    if (m_fOutput != NULL) {
      fwrite(lockBitstreamData.bitstreamBufferPtr, 1, lockBitstreamData.bitstreamSizeInBytes, m_fOutput);
    }
    // get those packets!
    switch (encodeConfig.codec) {
      case NV_ENC_H264:
        ParseH264Bitstream(lockBitstreamData, dts, packets);
        break;
      case NV_ENC_HEVC:
        ParseHEVCBitstream(lockBitstreamData, dts, packets);
        break;
    }

    nvStatus = m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, pEncodeBuffer->stOutputBfr.hBitstreamBuffer);
  }

  return nvStatus;
}

void NvEncoder::ParseH264Bitstream(NV_ENC_LOCK_BITSTREAM& lockBitstreamData, mtime_t& dts,
                                   std::vector<VideoStitch::IO::DataPacket>& packets) {
  std::vector<x264_nal_t> nalOut;
  x264_nal_t nal;
  switch (lockBitstreamData.pictureType) {
    case NV_ENC_PIC_TYPE_IDR:
      nal.i_type = NAL_SLICE_IDR;
      nal.i_ref_idc = NAL_PRIORITY_HIGHEST;
      break;
    case NV_ENC_PIC_TYPE_I:
    case NV_ENC_PIC_TYPE_P:
      nal.i_type = NAL_SLICE;
      nal.i_ref_idc = NAL_PRIORITY_HIGH;
      break;
    case NV_ENC_PIC_TYPE_B:
      nal.i_type = NAL_SLICE;
      nal.i_ref_idc = NAL_PRIORITY_LOW;
      break;
    default:
      break;
  }
  nal.p_payload = (uint8_t*)lockBitstreamData.bitstreamBufferPtr;
  nal.i_payload = int(lockBitstreamData.bitstreamSizeInBytes);
  nalOut.push_back(nal);

  if (dts > (mtime_t)lockBitstreamData.outputTimeStamp) {
    Logger::error(__NVENC__) << "dts " << dts << " value has to be lower than pts " << lockBitstreamData.outputTimeStamp
                             << std::endl;
    dts = lockBitstreamData.outputTimeStamp;
  }

  VideoEncoder::createDataPacket(nalOut, packets, mtime_t(std::round(lockBitstreamData.outputTimeStamp / 1000.0)),
                                 mtime_t(std::round(dts / 1000.0)));
}

void NvEncoder::ParseHEVCBitstream(NV_ENC_LOCK_BITSTREAM& lockBitstreamData, mtime_t& dts,
                                   std::vector<VideoStitch::IO::DataPacket>& packets) {
  std::vector<x265_nal> nalOut;
  uint8_t* start = (uint8_t*)lockBitstreamData.bitstreamBufferPtr;
  uint8_t* end = start + lockBitstreamData.bitstreamSizeInBytes;
  const static uint8_t start_seq[] = {0, 0, 1};
  start = std::search(start, end, start_seq, start_seq + 3);
  while (start != end) {
    decltype(start) next = std::search(start + 1, end, start_seq, start_seq + 3);
    x265_nal nal;
    nal.type = start[3] >> 1;
    nal.payload = start + 3;
    nal.sizeBytes = int(next - start - 3);
    nalOut.push_back(nal);
    start = next;
  }

  if (dts > (mtime_t)lockBitstreamData.outputTimeStamp) {
    Logger::error(__NVENC__) << "dts " << dts << " value has to be lower than pts " << lockBitstreamData.outputTimeStamp
                             << std::endl;
    dts = lockBitstreamData.outputTimeStamp;
  }

  VideoEncoder::createHEVCPacket(nalOut, packets, mtime_t(std::round(lockBitstreamData.outputTimeStamp / 1000)),
                                 mtime_t(std::round(dts / 1000)));
}

NVENCSTATUS NvEncoder::FlushEncoder() {
  if (!m_bEncoderInitialized) {
    return NV_ENC_SUCCESS;
  }
  NVENCSTATUS nvStatus = NvEncFlushEncoderQueue(m_stEOSOutputBfr.hOutputEvent);
  CHECK_RETURN(NvEncFlushEncoderQueue)

  EncodeBuffer* pEncodeBuffer = m_EncodeBufferQueue.GetPending();
  while (pEncodeBuffer) {
    // UnMap the input buffer after frame is done
    if (pEncodeBuffer && pEncodeBuffer->stInputBfr.hInputSurface) {
      nvStatus = NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
      CHECK_RETURN(NvEncUnmapInputResource)
      pEncodeBuffer->stInputBfr.hInputSurface = NULL;
    }
    pEncodeBuffer = m_EncodeBufferQueue.GetPending();
  }
#if defined(_WIN32)
  if (WaitForSingleObject(m_stEOSOutputBfr.hOutputEvent, 500) != WAIT_OBJECT_0) {
    Logger::error(__NVENC__) << "WaitForSingleObject timed out" << std::endl;
    nvStatus = NV_ENC_ERR_GENERIC;
  }
#endif
  return nvStatus;
}

Status NvEncoder::Initialize() {
  NVENCSTATUS nvStatus = InitCuda(encodeConfig.deviceID);
  if (nvStatus != NV_ENC_SUCCESS) {
    std::stringstream msg;
    msg << "InitCuda failed with code " << nvStatus;
    return Status(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__, msg.str());
  }

  CUcontext ctx;
  cuCtxGetCurrent(&ctx);
  nvStatus = InitializeAPI();
  if (nvStatus != NV_ENC_SUCCESS) {
    std::stringstream msg;
    msg << "Initialize failed with code " << nvStatus;
    return Status(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__, msg.str());
  }

  nvStatus = NvEncOpenEncodeSessionEx((void*)ctx, NV_ENC_DEVICE_TYPE_CUDA);
  if (nvStatus == NV_ENC_ERR_OUT_OF_MEMORY) {
    return Status(VideoStitch::Origin::Output, VideoStitch::ErrType::OutOfResources, __NVENC__, "Open session failed");
  } else if (nvStatus != NV_ENC_SUCCESS) {
    std::stringstream msg;
    msg << "Open session failed with code " << nvStatus;
    return Status(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__, msg.str());
  }

  encodeConfig.presetGUID = GetPresetGUID(encodeConfig.encoderPreset.c_str(), encodeConfig.codec);
  encodeConfig.profileGUID = GetProfileGUID(encodeConfig.encoderProfile.c_str(), encodeConfig.codec);
  if (encodeConfig.profileGUID == NV_ENC_H264_PROFILE_BASELINE_GUID) {
    /* There is no B-frames in H264 Baseline profile */
    Logger::info(__NVENC__) << "B-frames not supported in Baseline profile, discarding b_frames parameters"
                            << std::endl;
    encodeConfig.numB = 0;
  }
  if (encodeConfig.codec == NV_ENC_HEVC) {
    /* There is no B-frames in HEVC currently (Pascal generation) */
    Logger::info(__NVENC__) << "B-frames not supported in HEVC codec, discarding b_frames parameters" << std::endl;
    encodeConfig.numB = 0;
  }

  Logger::info(__NVENC__) << "Encoding codec           : " << (encodeConfig.codec == NV_ENC_HEVC ? "HEVC" : "H264")
                          << std::endl;
  Logger::info(__NVENC__) << "         size            : " << encodeConfig.width << "x" << encodeConfig.height
                          << std::endl;
  Logger::info(__NVENC__) << "         bitrate         : " << encodeConfig.bitrate << " kbits/sec" << std::endl;
  Logger::info(__NVENC__) << "         vbvMaxBitrate   : " << encodeConfig.vbvMaxBitrate << " kbits/sec" << std::endl;
  Logger::info(__NVENC__) << "         vbvSize         : " << encodeConfig.vbvSize << " kbits" << std::endl;
  Logger::info(__NVENC__) << "         fps             : " << encodeConfig.fps << " frames/sec" << std::endl;
  Logger::info(__NVENC__) << "         rcMode          : "
                          << (encodeConfig.rcMode == NV_ENC_PARAMS_RC_CONSTQP
                                  ? "CONSTQP"
                                  : encodeConfig.rcMode == NV_ENC_PARAMS_RC_VBR
                                        ? "VBR"
                                        : encodeConfig.rcMode == NV_ENC_PARAMS_RC_CBR
                                              ? "CBR"
                                              : encodeConfig.rcMode == NV_ENC_PARAMS_RC_VBR_MINQP
                                                    ? "VBR MINQP"
                                                    : encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_QUALITY
                                                          ? "TWO_PASS_QUALITY"
                                                          : encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP
                                                                ? "TWO_PASS_FRAMESIZE_CAP"
                                                                : encodeConfig.rcMode == NV_ENC_PARAMS_RC_2_PASS_VBR
                                                                      ? "TWO_PASS_VBR"
                                                                      : "UNKNOWN")
                          << std::endl;
  Logger::info(__NVENC__) << "         B frames        : " << encodeConfig.numB << std::endl;
  Logger::info(__NVENC__) << "         QP              : " << encodeConfig.qp << std::endl;
  Logger::info(__NVENC__)
      << "         preset          : "
      << ((encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HQ_GUID)
              ? "LOW_LATENCY_HQ"
              : (encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HP_GUID)
                    ? "LOW_LATENCY_HP"
                    : (encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID)
                          ? "LOW_LATENCY"
                          : (encodeConfig.presetGUID == NV_ENC_PRESET_HQ_GUID)
                                ? "HQ"
                                : (encodeConfig.presetGUID == NV_ENC_PRESET_HP_GUID)
                                      ? "HP"
                                      : (encodeConfig.presetGUID == NV_ENC_PRESET_BD_GUID)
                                            ? "BD"
                                            : (encodeConfig.presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID)
                                                  ? "LOSSLESS_HP"
                                                  : (encodeConfig.presetGUID == NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID)
                                                        ? "LOSSLESS"
                                                        : (encodeConfig.presetGUID == NV_ENC_PRESET_DEFAULT_GUID)
                                                              ? "DEFAULT"
                                                              : "UNKNOWN")
      << std::endl;
  Logger::info(__NVENC__)
      << "         profile         : "
      << ((encodeConfig.profileGUID == NV_ENC_H264_PROFILE_BASELINE_GUID)
              ? "BASELINE"
              : (encodeConfig.profileGUID == NV_ENC_H264_PROFILE_MAIN_GUID)
                    ? "MAIN"
                    : (encodeConfig.profileGUID == NV_ENC_HEVC_PROFILE_MAIN_GUID)
                          ? "MAIN"
                          : (encodeConfig.profileGUID == NV_ENC_H264_PROFILE_HIGH_GUID)
                                ? "HIGH"
                                : (encodeConfig.profileGUID == NV_ENC_H264_PROFILE_HIGH_444_GUID)
                                      ? "HIGH444"
                                      : (encodeConfig.profileGUID == NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID)
                                            ? "CONSTRAINED_HIGH"
                                            : (encodeConfig.profileGUID == NV_ENC_H264_PROFILE_STEREO_GUID)
                                                  ? "STEREO"
                                                  : (encodeConfig.profileGUID == NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID)
                                                        ? "AUTO"
                                                        : "UNKNOWN")
      << std::endl;
  nvStatus = CreateEncoder(&encodeConfig);
  if (nvStatus != NV_ENC_SUCCESS) {
    std::stringstream msg;
    msg << "CreateEncoder failed with code " << nvStatus;
    return Status(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__, msg.str());
  }

  /* update from sdk 6.0 samples */
  encodeConfig.maxWidth = encodeConfig.maxWidth ? encodeConfig.maxWidth : encodeConfig.width;
  encodeConfig.maxHeight = encodeConfig.maxHeight ? encodeConfig.maxHeight : encodeConfig.height;

  if ((encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HQ_GUID) ||
      (encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HP_GUID) ||
      (encodeConfig.presetGUID == NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID)) {
    m_uEncodeBufferCount = encodeConfig.numB + 2;
  } else if (encodeConfig.numB > 0) {
    m_uEncodeBufferCount = encodeConfig.numB + 4;  // min buffers is numb + 1 + 3 pipelining
  } else {
    int numMBs = ((encodeConfig.maxHeight + 15) >> 4) * ((encodeConfig.maxWidth + 15) >> 4);
    int NumIOBuffers;
    if (numMBs >= 32768) {  // 4kx2k
      NumIOBuffers = MAX_ENCODE_QUEUE / 8;
    } else if (numMBs >= 16384) {  // 2kx2k
      NumIOBuffers = MAX_ENCODE_QUEUE / 4;
    } else if (numMBs >= 8160) {  // 1920x1080
      NumIOBuffers = MAX_ENCODE_QUEUE / 2;
    } else {
      NumIOBuffers = MAX_ENCODE_QUEUE;
    }
    m_uEncodeBufferCount = NumIOBuffers;
  }

  nvStatus = AllocateIOBuffers(encodeConfig.width, encodeConfig.height);
  if (nvStatus != NV_ENC_SUCCESS) {
    std::stringstream msg;
    msg << "AllocateIOBuffers failed with code " << nvStatus;
    return Status(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__, msg.str());
  } else {
    return Status::OK();
  }
}

NVENCSTATUS NvEncoder::InitCuda(unsigned int deviceID) {
  int deviceCount = 0;

  cudaError_t r = cudaGetDeviceCount(&deviceCount);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::error(__NVENC__) << "cudaGetDeviceCount has returned CUDA error " << r << std::endl;
  }
  if (deviceCount == 0) {
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }

  if (deviceID > (unsigned int)deviceCount - 1) {
    Logger::error(__NVENC__) << "Invalid Device Id = " << deviceID << std::endl;
    return NV_ENC_ERR_INVALID_ENCODERDEVICE;
  }

  // Now we set the actual device
  r = cudaSetDevice(deviceID);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::error(__NVENC__) << "cudaSetDevice has returned CUDA error " << r << std::endl;
  }

  struct cudaDeviceProp props;
  r = cudaGetDeviceProperties(&props, deviceID);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::error(__NVENC__) << "cudaGetDeviceProperties has returned CUDA error " << r << std::endl;
  }
  if (((props.major << 4) + props.minor) < 0x30) {
    Logger::error(__NVENC__) << "GPU " << deviceID << " does not have NVENC capabilities exiting" << std::endl;
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }

  return NV_ENC_SUCCESS;
}

NVENCSTATUS NvEncoder::AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

  m_EncodeBufferQueue.Initialize(m_stEncodeBuffer, m_uEncodeBufferCount);

  for (uint32_t i = 0; i < m_uEncodeBufferCount; i++) {
    cudaError_t r = cudaMallocPitch(&m_stEncodeBuffer[i].stInputBfr.pNV12devPtr,
                                    (size_t*)&m_stEncodeBuffer[i].stInputBfr.uNV12Stride, encodeConfig.width,
                                    encodeConfig.height * 3 / 2);
    if (r != cudaSuccess) {
      resetCudaError();
      Logger::error(__NVENC__) << "cudaMallocPitch has returned CUDA error " << r << std::endl;
      return NV_ENC_ERR_OUT_OF_MEMORY;
    }

    m_stEncodeBuffer[i].stInputBfr.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
    m_stEncodeBuffer[i].stInputBfr.dwWidth = uInputWidth;
    m_stEncodeBuffer[i].stInputBfr.dwHeight = uInputHeight;

    nvStatus = NvEncRegisterResource(
        NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, (void*)m_stEncodeBuffer[i].stInputBfr.pNV12devPtr, uInputWidth,
        uInputHeight, m_stEncodeBuffer[i].stInputBfr.uNV12Stride, &m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource);
    CHECK_RETURN(NvEncRegisterResource)

    nvStatus = NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE, &m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
    CHECK_RETURN(NvEncCreateBitstreamBuffer)

    m_stEncodeBuffer[i].stOutputBfr.dwBitstreamBufferSize = BITSTREAM_BUFFER_SIZE;

#if defined(_WIN32)
    nvStatus = NvEncRegisterAsyncEvent(&m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
    CHECK_RETURN(NvEncRegisterAsyncEvent)
    m_stEncodeBuffer[i].stOutputBfr.bWaitOnEvent = true;
#else
    m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
#endif
  }

  m_stEOSOutputBfr.bEOSFlag = TRUE;
#if defined(_WIN32)
  nvStatus = NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.hOutputEvent);
  CHECK_RETURN(NvEncRegisterAsyncEvent)
#else
  m_stEOSOutputBfr.hOutputEvent = NULL;
#endif

  return NV_ENC_SUCCESS;
}

NVENCSTATUS NvEncoder::ReleaseIOBuffers() {
  for (uint32_t i = 0; i < m_uEncodeBufferCount; i++) {
    NVENCSTATUS nvStatus = NvEncDestroyBitstreamBuffer(m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
    CHECK_RETURN(NvEncDestroyBitstreamBuffer)
    m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer = NULL;

    nvStatus = NvEncUnregisterResource(m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource);
    CHECK_RETURN(NvEncUnregisterResource)
    m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource = NULL;

    cudaError_t r = cudaFree(m_stEncodeBuffer[i].stInputBfr.pNV12devPtr);
    if (r != cudaSuccess) {
      resetCudaError();
      Logger::error(__NVENC__) << "cudaFree has returned CUDA error " << r << std::endl;
      return NV_ENC_ERR_GENERIC;
    }
    m_stEncodeBuffer[i].stInputBfr.pNV12devPtr = NULL;

#if defined(_WIN32)
    NvEncUnregisterAsyncEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
    nvCloseFile(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
    m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
#endif
  }

  if (m_stEOSOutputBfr.hOutputEvent) {
#if defined(_WIN32)
    NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.hOutputEvent);
    nvCloseFile(m_stEOSOutputBfr.hOutputEvent);
    m_stEOSOutputBfr.hOutputEvent = NULL;
#endif
  }

  return NV_ENC_SUCCESS;
}

NVENCSTATUS NvEncoder::NvEncOpenEncodeSession(void* device, uint32_t deviceType) {
  return m_pEncodeAPI->nvEncOpenEncodeSession(device, deviceType, &m_hEncoder);
}

NVENCSTATUS NvEncoder::NvEncGetEncodeGUIDCount(uint32_t* encodeGUIDCount) {
  return m_pEncodeAPI->nvEncGetEncodeGUIDCount(m_hEncoder, encodeGUIDCount);
}

NVENCSTATUS NvEncoder::NvEncGetEncodeProfileGUIDCount(GUID encodeGUID, uint32_t* encodeProfileGUIDCount) {
  return m_pEncodeAPI->nvEncGetEncodeProfileGUIDCount(m_hEncoder, encodeGUID, encodeProfileGUIDCount);
}

NVENCSTATUS NvEncoder::NvEncGetEncodeProfileGUIDs(GUID encodeGUID, GUID* profileGUIDs, uint32_t guidArraySize,
                                                  uint32_t* GUIDCount) {
  return m_pEncodeAPI->nvEncGetEncodeProfileGUIDs(m_hEncoder, encodeGUID, profileGUIDs, guidArraySize, GUIDCount);
}

NVENCSTATUS NvEncoder::NvEncGetEncodeGUIDs(GUID* GUIDs, uint32_t guidArraySize, uint32_t* GUIDCount) {
  return m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, GUIDs, guidArraySize, GUIDCount);
}

NVENCSTATUS NvEncoder::NvEncGetInputFormatCount(GUID encodeGUID, uint32_t* inputFmtCount) {
  return m_pEncodeAPI->nvEncGetInputFormatCount(m_hEncoder, encodeGUID, inputFmtCount);
}

NVENCSTATUS NvEncoder::NvEncGetInputFormats(GUID encodeGUID, NV_ENC_BUFFER_FORMAT* inputFmts,
                                            uint32_t inputFmtArraySize, uint32_t* inputFmtCount) {
  return m_pEncodeAPI->nvEncGetInputFormats(m_hEncoder, encodeGUID, inputFmts, inputFmtArraySize, inputFmtCount);
}

NVENCSTATUS NvEncoder::NvEncGetEncodeCaps(GUID encodeGUID, NV_ENC_CAPS_PARAM* capsParam, int* capsVal) {
  return m_pEncodeAPI->nvEncGetEncodeCaps(m_hEncoder, encodeGUID, capsParam, capsVal);
}

NVENCSTATUS NvEncoder::NvEncGetEncodePresetCount(GUID encodeGUID, uint32_t* encodePresetGUIDCount) {
  return m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, encodeGUID, encodePresetGUIDCount);
}

NVENCSTATUS NvEncoder::NvEncGetEncodePresetGUIDs(GUID encodeGUID, GUID* presetGUIDs, uint32_t guidArraySize,
                                                 uint32_t* encodePresetGUIDCount) {
  return m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, encodeGUID, presetGUIDs, guidArraySize,
                                                 encodePresetGUIDCount);
}

NVENCSTATUS NvEncoder::NvEncGetEncodePresetConfig(GUID encodeGUID, GUID presetGUID,
                                                  NV_ENC_PRESET_CONFIG* presetConfig) {
  return m_pEncodeAPI->nvEncGetEncodePresetConfig(m_hEncoder, encodeGUID, presetGUID, presetConfig);
}

NVENCSTATUS NvEncoder::NvEncCreateInputBuffer(uint32_t width, uint32_t height, void** inputBuffer, uint32_t isYuv444) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
  NV_ENC_CREATE_INPUT_BUFFER createInputBufferParams;

  memset(&createInputBufferParams, 0, sizeof(createInputBufferParams));
  SET_VER(createInputBufferParams, NV_ENC_CREATE_INPUT_BUFFER);

  createInputBufferParams.width = width;
  createInputBufferParams.height = height;
  createInputBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
  createInputBufferParams.bufferFmt = isYuv444 ? NV_ENC_BUFFER_FORMAT_YUV444_PL : NV_ENC_BUFFER_FORMAT_NV12_PL;

  nvStatus = m_pEncodeAPI->nvEncCreateInputBuffer(m_hEncoder, &createInputBufferParams);
  CHECK(NvEncFlushEncoderQueue)

  *inputBuffer = createInputBufferParams.inputBuffer;

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer) {
  if (inputBuffer) {
    return m_pEncodeAPI->nvEncDestroyInputBuffer(m_hEncoder, inputBuffer);
  }
  return NV_ENC_SUCCESS;
}

NVENCSTATUS NvEncoder::NvEncCreateBitstreamBuffer(uint32_t size, void** bitstreamBuffer) {
  NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams;

  memset(&createBitstreamBufferParams, 0, sizeof(createBitstreamBufferParams));
  SET_VER(createBitstreamBufferParams, NV_ENC_CREATE_BITSTREAM_BUFFER);

  createBitstreamBufferParams.size = size;
  createBitstreamBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

  NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncCreateBitstreamBuffer(m_hEncoder, &createBitstreamBufferParams);
  CHECK(nvEncCreateBitstreamBuffer)

  *bitstreamBuffer = createBitstreamBufferParams.bitstreamBuffer;

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

  if (bitstreamBuffer) {
    nvStatus = m_pEncodeAPI->nvEncDestroyBitstreamBuffer(m_hEncoder, bitstreamBuffer);
    CHECK(nvEncDestroyBitstreamBuffer)
  }

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM* lockBitstreamBufferParams) {
  return m_pEncodeAPI->nvEncLockBitstream(m_hEncoder, lockBitstreamBufferParams);
}

NVENCSTATUS NvEncoder::NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer) {
  return m_pEncodeAPI->nvEncUnlockBitstream(m_hEncoder, bitstreamBuffer);
}

NVENCSTATUS NvEncoder::NvEncLockInputBuffer(void* inputBuffer, void** bufferDataPtr, uint32_t* pitch) {
  NV_ENC_LOCK_INPUT_BUFFER lockInputBufferParams;

  memset(&lockInputBufferParams, 0, sizeof(lockInputBufferParams));
  SET_VER(lockInputBufferParams, NV_ENC_LOCK_INPUT_BUFFER);

  lockInputBufferParams.inputBuffer = inputBuffer;
  NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncLockInputBuffer(m_hEncoder, &lockInputBufferParams);
  CHECK(nvEncLockInputBuffer)

  *bufferDataPtr = lockInputBufferParams.bufferDataPtr;
  *pitch = lockInputBufferParams.pitch;

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer) {
  return m_pEncodeAPI->nvEncUnlockInputBuffer(m_hEncoder, inputBuffer);
}

NVENCSTATUS NvEncoder::NvEncGetEncodeStats(NV_ENC_STAT* encodeStats) {
  return m_pEncodeAPI->nvEncGetEncodeStats(m_hEncoder, encodeStats);
}

NVENCSTATUS NvEncoder::NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload) {
  return m_pEncodeAPI->nvEncGetSequenceParams(m_hEncoder, sequenceParamPayload);
}

NVENCSTATUS NvEncoder::NvEncRegisterAsyncEvent(void** completionEvent) {
  NV_ENC_EVENT_PARAMS eventParams;

  memset(&eventParams, 0, sizeof(eventParams));
  SET_VER(eventParams, NV_ENC_EVENT_PARAMS);

#if defined(_WIN32)
  eventParams.completionEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
#else
  eventParams.completionEvent = NULL;
#endif
  NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncRegisterAsyncEvent(m_hEncoder, &eventParams);
  CHECK(nvEncRegisterAsyncEvent)

  *completionEvent = eventParams.completionEvent;

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncUnregisterAsyncEvent(void* completionEvent) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
  NV_ENC_EVENT_PARAMS eventParams;

  if (completionEvent) {
    memset(&eventParams, 0, sizeof(eventParams));
    SET_VER(eventParams, NV_ENC_EVENT_PARAMS);

    eventParams.completionEvent = completionEvent;

    nvStatus = m_pEncodeAPI->nvEncUnregisterAsyncEvent(m_hEncoder, &eventParams);
    CHECK(nvEncUnregisterAsyncEvent)
  }

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncMapInputResource(void* registeredResource, void** mappedResource) {
  NV_ENC_MAP_INPUT_RESOURCE mapInputResParams;

  memset(&mapInputResParams, 0, sizeof(mapInputResParams));
  SET_VER(mapInputResParams, NV_ENC_MAP_INPUT_RESOURCE);

  mapInputResParams.registeredResource = registeredResource;

  NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncMapInputResource(m_hEncoder, &mapInputResParams);
  CHECK(nvEncMapInputResource)

  *mappedResource = mapInputResParams.mappedResource;

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

  if (mappedInputBuffer) {
    nvStatus = m_pEncodeAPI->nvEncUnmapInputResource(m_hEncoder, mappedInputBuffer);
    CHECK(nvEncUnmapInputResource)
  }

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncDestroyEncoder() {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

  if (m_hEncoder) {
    nvStatus = m_pEncodeAPI->nvEncDestroyEncoder(m_hEncoder);
    CHECK(nvEncDestroyEncoder)
  }
  m_bEncoderInitialized = false;

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncInvalidateRefFrames(const NvEncPictureCommand* pEncPicCommand) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

  for (uint32_t i = 0; i < pEncPicCommand->numRefFramesToInvalidate; i++) {
    nvStatus = m_pEncodeAPI->nvEncInvalidateRefFrames(m_hEncoder, pEncPicCommand->refFrameNumbers[i]);
    CHECK(nvEncInvalidateRefFrames)
  }

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncOpenEncodeSessionEx(void* device, NV_ENC_DEVICE_TYPE deviceType) {
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS openSessionExParams;

  memset(&openSessionExParams, 0, sizeof(openSessionExParams));
  SET_VER(openSessionExParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);

  openSessionExParams.device = device;
  openSessionExParams.deviceType = deviceType;
  openSessionExParams.reserved = NULL;
  openSessionExParams.apiVersion = NVENCAPI_VERSION;

  return m_pEncodeAPI->nvEncOpenEncodeSessionEx(&openSessionExParams, &m_hEncoder);
}

NVENCSTATUS NvEncoder::NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister,
                                             uint32_t width, uint32_t height, uint32_t pitch,
                                             void** registeredResource) {
  NV_ENC_REGISTER_RESOURCE registerResParams;

  memset(&registerResParams, 0, sizeof(registerResParams));
  SET_VER(registerResParams, NV_ENC_REGISTER_RESOURCE);

  registerResParams.resourceType = resourceType;
  registerResParams.resourceToRegister = resourceToRegister;
  registerResParams.width = width;
  registerResParams.height = height;
  registerResParams.pitch = pitch;
  registerResParams.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12_PL;

  NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncRegisterResource(m_hEncoder, &registerResParams);
  CHECK(nvEncRegisterResource)

  *registeredResource = registerResParams.registeredResource;

  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes) {
  return m_pEncodeAPI->nvEncUnregisterResource(m_hEncoder, registeredRes);
}

NVENCSTATUS NvEncoder::ValidateEncodeGUID(GUID inputCodecGuid) {
  unsigned int i, codecFound, encodeGUIDCount, encodeGUIDArraySize;
  NVENCSTATUS nvStatus;
  GUID* encodeGUIDArray;

  nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDCount(m_hEncoder, &encodeGUIDCount);
  CHECK_RETURN(nvEncGetEncodeGUIDCount)

  encodeGUIDArray = new GUID[encodeGUIDCount];
  memset(encodeGUIDArray, 0, sizeof(GUID) * encodeGUIDCount);

  encodeGUIDArraySize = 0;
  nvStatus = m_pEncodeAPI->nvEncGetEncodeGUIDs(m_hEncoder, encodeGUIDArray, encodeGUIDCount, &encodeGUIDArraySize);
  CHECK(nvEncGetEncodeGUIDs)
  if (nvStatus != NV_ENC_SUCCESS) {
    delete[] encodeGUIDArray;
    return nvStatus;
  }

  if (encodeGUIDArraySize > encodeGUIDCount) {
    Logger::warning(__NVENC__) << "encodeGUIDArraySize (" << encodeGUIDArraySize << ") > (" << encodeGUIDCount
                               << ") encodeGUIDCount" << std::endl;
  }

  codecFound = 0;
  for (i = 0; i < encodeGUIDArraySize; i++) {
    if (inputCodecGuid == encodeGUIDArray[i]) {
      codecFound = 1;
      break;
    }
  }

  delete[] encodeGUIDArray;

  if (codecFound) {
    return NV_ENC_SUCCESS;
  } else {
    return NV_ENC_ERR_INVALID_PARAM;
  }
}

NVENCSTATUS NvEncoder::ValidatePresetGUID(GUID inputPresetGuid, GUID inputCodecGuid) {
  uint32_t i, presetFound, presetGUIDCount, presetGUIDArraySize;
  NVENCSTATUS nvStatus;
  GUID* presetGUIDArray;

  nvStatus = m_pEncodeAPI->nvEncGetEncodePresetCount(m_hEncoder, inputCodecGuid, &presetGUIDCount);
  CHECK_RETURN(nvEncGetEncodePresetCount)

  presetGUIDArray = new GUID[presetGUIDCount];
  memset(presetGUIDArray, 0, sizeof(GUID) * presetGUIDCount);

  presetGUIDArraySize = 0;
  nvStatus = m_pEncodeAPI->nvEncGetEncodePresetGUIDs(m_hEncoder, inputCodecGuid, presetGUIDArray, presetGUIDCount,
                                                     &presetGUIDArraySize);
  CHECK(nvEncGetEncodePresetGUIDs)
  if (nvStatus != NV_ENC_SUCCESS) {
    delete[] presetGUIDArray;
    return nvStatus;
  }

  if (presetGUIDArraySize > presetGUIDCount) {
    Logger::warning(__NVENC__) << "presetGUIDArraySize (" << presetGUIDArraySize << ") > (" << presetGUIDCount
                               << ") presetGUIDCount" << std::endl;
  }

  presetFound = 0;
  for (i = 0; i < presetGUIDArraySize; i++) {
    if (inputPresetGuid == presetGUIDArray[i]) {
      presetFound = 1;
      break;
    }
  }

  delete[] presetGUIDArray;

  if (presetFound) {
    return NV_ENC_SUCCESS;
  } else {
    return NV_ENC_ERR_INVALID_PARAM;
  }
}

NVENCSTATUS NvEncoder::ValidateProfileGUID(GUID inputProfileGuid, GUID inputCodecGuid) {
  uint32_t i, profileFound, profileGUIDCount, profileGUIDArraySize;
  NVENCSTATUS nvStatus;
  GUID* profileGUIDArray;

  nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDCount(m_hEncoder, inputCodecGuid, &profileGUIDCount);
  CHECK_RETURN(nvEncGetEncodeProfileGUIDCount)

  profileGUIDArray = new GUID[profileGUIDCount];
  memset(profileGUIDArray, 0, sizeof(GUID) * profileGUIDCount);

  profileGUIDArraySize = 0;
  nvStatus = m_pEncodeAPI->nvEncGetEncodeProfileGUIDs(m_hEncoder, inputCodecGuid, profileGUIDArray, profileGUIDCount,
                                                      &profileGUIDArraySize);
  CHECK(nvEncGetEncodeProfileGUIDs)
  if (nvStatus != NV_ENC_SUCCESS) {
    delete[] profileGUIDArray;
    return nvStatus;
  }

  if (profileGUIDArraySize > profileGUIDCount) {
    Logger::warning(__NVENC__) << "profileGUIDArraySize (" << profileGUIDArraySize << ") > (" << profileGUIDCount
                               << ") profileGUIDCount" << std::endl;
  }

  profileFound = 0;
  for (i = 0; i < profileGUIDArraySize; i++) {
    if (inputProfileGuid == profileGUIDArray[i]) {
      profileFound = 1;
      break;
    }
  }

  delete[] profileGUIDArray;

  if (profileFound) {
    return NV_ENC_SUCCESS;
  } else {
    return NV_ENC_ERR_INVALID_PARAM;
  }
}

NVENCSTATUS NvEncoder::CreateEncoder(const EncodeConfig* pEncCfg) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

  if (pEncCfg == NULL) {
    return NV_ENC_ERR_INVALID_PARAM;
  }

  int curWidth = pEncCfg->width;
  int curHeight = pEncCfg->height;

  int maxWidth = (pEncCfg->maxWidth > 0 ? pEncCfg->maxWidth : pEncCfg->width);
  int maxHeight = (pEncCfg->maxHeight > 0 ? pEncCfg->maxHeight : pEncCfg->height);

  if ((curWidth > maxWidth) || (curHeight > maxHeight)) {
    return NV_ENC_ERR_INVALID_PARAM;
  }

  if (!pEncCfg->width || !pEncCfg->height) {
    return NV_ENC_ERR_INVALID_PARAM;
  }

  GUID inputCodecGUID = pEncCfg->codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
  nvStatus = ValidateEncodeGUID(inputCodecGUID);
  CHECK_RETURN(ValidateEncodeGUID)

  codecGUID = inputCodecGUID;

  m_stCreateEncodeParams.encodeGUID = inputCodecGUID;
  m_stCreateEncodeParams.presetGUID = pEncCfg->presetGUID;
  m_stCreateEncodeParams.encodeWidth = pEncCfg->width;
  m_stCreateEncodeParams.encodeHeight = pEncCfg->height;

  m_stCreateEncodeParams.darWidth = pEncCfg->width;
  m_stCreateEncodeParams.darHeight = pEncCfg->height;
  m_stCreateEncodeParams.frameRateNum = pEncCfg->fps.num;
  m_stCreateEncodeParams.frameRateDen = pEncCfg->fps.den;
#if defined(_WIN32)
  m_stCreateEncodeParams.enableEncodeAsync = 1;
#else
  m_stCreateEncodeParams.enableEncodeAsync = 0;
#endif
  m_stCreateEncodeParams.enablePTD = 1;
  m_stCreateEncodeParams.reportSliceOffsets = 0;
  m_stCreateEncodeParams.enableSubFrameWrite = 0;
  m_stCreateEncodeParams.encodeConfig = &m_stEncodeConfig;
  m_stCreateEncodeParams.maxEncodeWidth = maxWidth;
  m_stCreateEncodeParams.maxEncodeHeight = maxHeight;

  // apply preset
  NV_ENC_PRESET_CONFIG stPresetCfg;
  memset(&stPresetCfg, 0, sizeof(NV_ENC_PRESET_CONFIG));
  SET_VER(stPresetCfg, NV_ENC_PRESET_CONFIG);
  SET_VER(stPresetCfg.presetCfg, NV_ENC_CONFIG);

  nvStatus = m_pEncodeAPI->nvEncGetEncodePresetConfig(m_hEncoder, m_stCreateEncodeParams.encodeGUID,
                                                      m_stCreateEncodeParams.presetGUID, &stPresetCfg);
  CHECK_RETURN(nvEncGetEncodePresetConfig)

  memcpy(&m_stEncodeConfig, &stPresetCfg.presetCfg, sizeof(NV_ENC_CONFIG));

  if (pEncCfg->gopLength > 0) {
    m_stEncodeConfig.gopLength = pEncCfg->gopLength;
    m_stEncodeConfig.frameIntervalP = pEncCfg->numB + 1;
  } else if (pEncCfg->gopLength == 0) {
    m_stEncodeConfig.gopLength = 1;
    m_stEncodeConfig.frameIntervalP = 0;
  }
  if (m_stEncodeConfig.gopLength == NVENC_INFINITE_GOPLENGTH) {
    Logger::info(__NVENC__) << "         goplength       : INFINITE GOP" << std::endl;
  } else {
    Logger::info(__NVENC__) << "         goplength       : " << m_stEncodeConfig.gopLength << std::endl;
  }

  m_stEncodeConfig.profileGUID = pEncCfg->profileGUID;
  if (pEncCfg->pictureStruct == NV_ENC_PIC_STRUCT_FRAME) {
    m_stEncodeConfig.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
  } else {
    m_stEncodeConfig.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
  }

  m_stEncodeConfig.mvPrecision = NV_ENC_MV_PRECISION_QUARTER_PEL;

  if (pEncCfg->bitrate || pEncCfg->vbvMaxBitrate) {
    m_stEncodeConfig.rcParams.rateControlMode = (NV_ENC_PARAMS_RC_MODE)pEncCfg->rcMode;
    m_stEncodeConfig.rcParams.averageBitRate = 1000 * pEncCfg->bitrate;
    m_stEncodeConfig.rcParams.maxBitRate = 1000 * pEncCfg->vbvMaxBitrate;
    m_stEncodeConfig.rcParams.vbvBufferSize = 1000 * pEncCfg->vbvSize;
    m_stEncodeConfig.rcParams.vbvInitialDelay = 1000 * pEncCfg->vbvSize * 9 / 10;
    /* filler data is inserted if needed to achieve hrd bitrate in CBR */
    if (((m_stEncodeConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_CBR) ||
         (m_stEncodeConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_2_PASS_QUALITY) ||
         (m_stEncodeConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP))) {
      if (pEncCfg->codec == NV_ENC_H264) {
        m_stEncodeConfig.encodeCodecConfig.h264Config.outputBufferingPeriodSEI = 1;
        m_stEncodeConfig.encodeCodecConfig.h264Config.outputPictureTimingSEI = 1;
      } else {
        m_stEncodeConfig.encodeCodecConfig.hevcConfig.outputBufferingPeriodSEI = 1;
        m_stEncodeConfig.encodeCodecConfig.hevcConfig.outputPictureTimingSEI = 1;
      }
    }
  } else {
    m_stEncodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
  }

  if (pEncCfg->rcMode == 0) {
    m_stEncodeConfig.rcParams.constQP.qpInterP =
        pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : pEncCfg->qp;
    m_stEncodeConfig.rcParams.constQP.qpInterB =
        pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : pEncCfg->qp;
    m_stEncodeConfig.rcParams.constQP.qpIntra = pEncCfg->presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : pEncCfg->qp;
  }

  // set up initial QP value
  if (pEncCfg->rcMode == NV_ENC_PARAMS_RC_VBR || pEncCfg->rcMode == NV_ENC_PARAMS_RC_VBR_MINQP ||
      pEncCfg->rcMode == NV_ENC_PARAMS_RC_2_PASS_VBR) {
    m_stEncodeConfig.rcParams.enableInitialRCQP = 1;
    m_stEncodeConfig.rcParams.initialRCQP.qpInterP = pEncCfg->qp;
    if (pEncCfg->i_quant_factor != 0.0 && pEncCfg->b_quant_factor != 0.0) {
      m_stEncodeConfig.rcParams.initialRCQP.qpIntra =
          (int)(pEncCfg->qp * FABS(pEncCfg->i_quant_factor) + pEncCfg->i_quant_offset);
      m_stEncodeConfig.rcParams.initialRCQP.qpInterB =
          (int)(pEncCfg->qp * FABS(pEncCfg->b_quant_factor) + pEncCfg->b_quant_offset);
    } else {
      m_stEncodeConfig.rcParams.initialRCQP.qpIntra = pEncCfg->qp;
      m_stEncodeConfig.rcParams.initialRCQP.qpInterB = pEncCfg->qp;
    }
  }

  if (pEncCfg->intraRefreshEnableFlag) {
    if (pEncCfg->codec == NV_ENC_HEVC) {
      m_stEncodeConfig.encodeCodecConfig.hevcConfig.enableIntraRefresh = 1;
      m_stEncodeConfig.encodeCodecConfig.hevcConfig.intraRefreshPeriod = pEncCfg->intraRefreshPeriod;
      m_stEncodeConfig.encodeCodecConfig.hevcConfig.intraRefreshCnt = pEncCfg->intraRefreshDuration;
    } else {
      m_stEncodeConfig.encodeCodecConfig.h264Config.enableIntraRefresh = 1;
      m_stEncodeConfig.encodeCodecConfig.h264Config.intraRefreshPeriod = pEncCfg->intraRefreshPeriod;
      m_stEncodeConfig.encodeCodecConfig.h264Config.intraRefreshCnt = pEncCfg->intraRefreshDuration;
    }
  }

  if (pEncCfg->invalidateRefFramesEnableFlag) {
    if (pEncCfg->codec == NV_ENC_HEVC) {
      m_stEncodeConfig.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = 16;
    } else {
      m_stEncodeConfig.encodeCodecConfig.h264Config.maxNumRefFrames = 16;
    }
  }

  if (pEncCfg->qpDeltaMapFile) {
    m_stEncodeConfig.rcParams.qpMapMode = NV_ENC_QP_MAP_EMPHASIS;
  }
  if (pEncCfg->codec == NV_ENC_H264) {
    if (pEncCfg->gopLength >= 0) {
      m_stEncodeConfig.encodeCodecConfig.h264Config.idrPeriod = m_stEncodeConfig.gopLength;
    }
    m_stEncodeConfig.encodeCodecConfig.h264Config.sliceMode = 3;
    m_stEncodeConfig.encodeCodecConfig.h264Config.sliceModeData = 1;
    m_stEncodeConfig.encodeCodecConfig.h264Config.chromaFormatIDC = (pEncCfg->isYuv444) ? 3 : 1;
    m_stEncodeConfig.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
    m_stEncodeConfig.encodeCodecConfig.h264Config.outputAUD = 1;
  } else if (pEncCfg->codec == NV_ENC_HEVC) {
    if (pEncCfg->gopLength >= 0) {
      m_stEncodeConfig.encodeCodecConfig.hevcConfig.idrPeriod = m_stEncodeConfig.gopLength;
    }
    m_stEncodeConfig.encodeCodecConfig.hevcConfig.sliceMode = 3;
    m_stEncodeConfig.encodeCodecConfig.hevcConfig.sliceModeData = 1;
    m_stEncodeConfig.encodeCodecConfig.hevcConfig.chromaFormatIDC = (pEncCfg->isYuv444) ? 3 : 1;
    m_stEncodeConfig.encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;
    m_stEncodeConfig.encodeCodecConfig.hevcConfig.outputAUD = 1;
  }

  nvStatus = m_pEncodeAPI->nvEncInitializeEncoder(m_hEncoder, &m_stCreateEncodeParams);
  CHECK_RETURN(nvEncInitializeEncoder)

  m_bEncoderInitialized = true;

  /* get header for RTMP configuration  */
  uint32_t outSize = 0;
  NV_ENC_SEQUENCE_PARAM_PAYLOAD payload = {0};
  payload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;

  headPkt.resize(256);
  payload.spsppsBuffer = headPkt.data();
  payload.inBufferSize = (uint32_t)headPkt.size();
  payload.outSPSPPSPayloadSize = &outSize;
  nvStatus = NvEncGetSequenceParams(&payload);
  CHECK_RETURN(NvEncGetSequenceParams)
  headPkt.resize(outSize);

  return nvStatus;
}

GUID NvEncoder::GetPresetGUID(const char* encoderPreset, int codec) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
  GUID presetGUID = NV_ENC_PRESET_DEFAULT_GUID;

  if (encoderPreset && (stricmp(encoderPreset, "hq") == 0)) {
    presetGUID = NV_ENC_PRESET_HQ_GUID;
  } else if (encoderPreset && (stricmp(encoderPreset, "llhp") == 0)) {
    presetGUID = NV_ENC_PRESET_LOW_LATENCY_HP_GUID;
  } else if (encoderPreset && (stricmp(encoderPreset, "hp") == 0)) {
    presetGUID = NV_ENC_PRESET_HP_GUID;
  } else if (encoderPreset && (stricmp(encoderPreset, "losslesshp") == 0)) {
    presetGUID = NV_ENC_PRESET_LOSSLESS_HP_GUID;
  } else if (encoderPreset && (stricmp(encoderPreset, "llhq") == 0)) {
    presetGUID = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
  } else if (encoderPreset && (stricmp(encoderPreset, "bd") == 0)) {
    presetGUID = NV_ENC_PRESET_BD_GUID;
  } else if (encoderPreset && (stricmp(encoderPreset, "ll") == 0)) {
    presetGUID = NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID;
  } else if (encoderPreset && (stricmp(encoderPreset, "lossless") == 0)) {
    presetGUID = NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID;
  } else {
    if (encoderPreset && (stricmp(encoderPreset, "") != 0)) {
      Logger::warning(__NVENC__) << "Unsupported preset guid : " << encoderPreset << " => using default one."
                                 << std::endl;
    }
    presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
  }

  GUID inputCodecGUID = codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
  nvStatus = ValidatePresetGUID(presetGUID, inputCodecGUID);
  CHECK(ValidatePresetGUID)
  if (nvStatus != NV_ENC_SUCCESS) {
    presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
  }

  return presetGUID;
}

GUID NvEncoder::GetProfileGUID(const char* encoderProfile, int codec) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
  GUID profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;

  if (encoderProfile && (stricmp(encoderProfile, "baseline") == 0)) {
    profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;
  } else if (encoderProfile && (stricmp(encoderProfile, "main") == 0)) {
    profileGUID = (codec == NV_ENC_H264) ? NV_ENC_H264_PROFILE_MAIN_GUID : NV_ENC_HEVC_PROFILE_MAIN_GUID;
  } else if (encoderProfile && (stricmp(encoderProfile, "high") == 0)) {
    profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
  } else if (encoderProfile && (stricmp(encoderProfile, "constrained_high") == 0)) {
    profileGUID = NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID;
  } else if (encoderProfile && (stricmp(encoderProfile, "high444") == 0)) {
    profileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID;
  } else if (encoderProfile && (stricmp(encoderProfile, "stereo") == 0)) {
    profileGUID = NV_ENC_H264_PROFILE_STEREO_GUID;
  } else {
    if (encoderProfile && (stricmp(encoderProfile, "") != 0)) {
      Logger::error(__NVENC__) << "Unsupported profile guid " << encoderProfile << std::endl;
    }
    profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
  }

  GUID inputCodecGUID = codec == NV_ENC_H264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
  nvStatus = ValidateProfileGUID(profileGUID, inputCodecGUID);
  CHECK(ValidateProfileGUID)
  if (nvStatus != NV_ENC_SUCCESS) {
    profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
  }

  return profileGUID;
}

NVENCSTATUS NvEncoder::InitializeAPI() {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
  MYPROC nvEncodeAPICreateInstance;  // function pointer to create instance in nvEncodeAPI

#if defined(_WIN32)
#if defined(_WIN64)
  m_hinstLib = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
#else
  m_hinstLib = LoadLibrary(TEXT("nvEncodeAPI.dll"));
#endif
#else
  m_hinstLib = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
#endif
  if (m_hinstLib == NULL) {
    return NV_ENC_ERR_OUT_OF_MEMORY;
  }

#if defined(_WIN32)
  nvEncodeAPICreateInstance = (MYPROC)GetProcAddress(m_hinstLib, "NvEncodeAPICreateInstance");
#else
  nvEncodeAPICreateInstance = (MYPROC)dlsym(m_hinstLib, "NvEncodeAPICreateInstance");
#endif

  if (nvEncodeAPICreateInstance == NULL) {
    return NV_ENC_ERR_OUT_OF_MEMORY;
  }

  m_pEncodeAPI = new NV_ENCODE_API_FUNCTION_LIST;
  if (m_pEncodeAPI == NULL) {
    return NV_ENC_ERR_OUT_OF_MEMORY;
  }

  memset(m_pEncodeAPI, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
  m_pEncodeAPI->version = NV_ENCODE_API_FUNCTION_LIST_VER;
  nvStatus = nvEncodeAPICreateInstance(m_pEncodeAPI);
  CHECK_RETURN(nvEncodeAPICreateInstance)

  return NV_ENC_SUCCESS;
}

NVENCSTATUS NvEncoder::NvEncEncodeFrame(EncodeBuffer* pEncodeBuffer, NvEncPictureCommand* encPicCommand, uint32_t width,
                                        uint32_t height, mtime_t pts, NV_ENC_PIC_STRUCT ePicStruct,
                                        int8_t* qpDeltaMapArray, uint32_t qpDeltaMapArraySize) {
  NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
  NV_ENC_PIC_PARAMS encPicParams;

  memset(&encPicParams, 0, sizeof(encPicParams));
  SET_VER(encPicParams, NV_ENC_PIC_PARAMS);

  encPicParams.inputBuffer = pEncodeBuffer->stInputBfr.hInputSurface;
  encPicParams.bufferFmt = pEncodeBuffer->stInputBfr.bufferFmt;
  encPicParams.inputWidth = width;
  encPicParams.inputHeight = height;
  encPicParams.inputPitch = pEncodeBuffer->stInputBfr.uNV12Stride;
  encPicParams.outputBitstream = pEncodeBuffer->stOutputBfr.hBitstreamBuffer;
  encPicParams.completionEvent = pEncodeBuffer->stOutputBfr.hOutputEvent;
  encPicParams.inputTimeStamp = pts;
  encPicParams.pictureStruct = ePicStruct;
  encPicParams.qpDeltaMap = qpDeltaMapArray;
  encPicParams.qpDeltaMapSize = qpDeltaMapArraySize;

  if (encPicCommand) {
    if (encPicCommand->bForceIDR) {
      encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
    }

    if (encPicCommand->bForceIntraRefresh) {
      if (codecGUID == NV_ENC_CODEC_HEVC_GUID) {
        encPicParams.codecPicParams.hevcPicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
      } else {
        encPicParams.codecPicParams.h264PicParams.forceIntraRefreshWithFrameCnt = encPicCommand->intraRefreshDuration;
      }
    }
  }

  nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
  if (nvStatus != NV_ENC_SUCCESS && nvStatus != NV_ENC_ERR_NEED_MORE_INPUT) {
    Logger::warning(__NVENC__) << "nvEncEncodePicture failed with code " << nvStatus << std::endl;
    return nvStatus;
  }

  return NV_ENC_SUCCESS;
}

NVENCSTATUS NvEncoder::NvEncReconfigureEncoder(NV_ENC_RECONFIGURE_PARAMS* reInitEncodeParams) {
  NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncReconfigureEncoder(m_hEncoder, reInitEncodeParams);
  CHECK(NvEncReconfigureEncoder)
  return nvStatus;
}

NVENCSTATUS NvEncoder::NvEncFlushEncoderQueue(void* hEOSEvent) {
  NV_ENC_PIC_PARAMS encPicParams;
  memset(&encPicParams, 0, sizeof(encPicParams));
  SET_VER(encPicParams, NV_ENC_PIC_PARAMS);
  encPicParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
  encPicParams.completionEvent = hEOSEvent;
  NVENCSTATUS nvStatus = m_pEncodeAPI->nvEncEncodePicture(m_hEncoder, &encPicParams);
  CHECK(nvEncEncodePicture)

  return nvStatus;
}
}
}  // namespace Output
}  // namespace VideoStitch
