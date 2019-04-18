// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videoEncoder.hpp"
#include "amfIncludes.hpp"

#include <nvEncodeAPI.h>
#include <cuda_runtime.h>

#include <queue>

#define SET_VER(configStruct, type) \
  { configStruct.version = type##_VER; }
#define MAX_ENCODE_QUEUE 32
#define BITSTREAM_BUFFER_SIZE 2 * 1024 * 1024

#if defined __linux__
#include <dlfcn.h>
#include <limits.h>

#define FALSE 0
#define TRUE 1
#define stricmp strcasecmp
typedef void* HANDLE;
typedef void* HINSTANCE;

__inline bool operator==(const GUID& guid1, const GUID& guid2) {
  if (guid1.Data1 == guid2.Data1 && guid1.Data2 == guid2.Data2 && guid1.Data3 == guid2.Data3 &&
      guid1.Data4[0] == guid2.Data4[0] && guid1.Data4[1] == guid2.Data4[1] && guid1.Data4[2] == guid2.Data4[2] &&
      guid1.Data4[3] == guid2.Data4[3] && guid1.Data4[4] == guid2.Data4[4] && guid1.Data4[5] == guid2.Data4[5] &&
      guid1.Data4[6] == guid2.Data4[6] && guid1.Data4[7] == guid2.Data4[7]) {
    return true;
  }

  return false;
}
#elif defined(_WIN32) || defined(_WIN64)
// The POSIX name for stricmp is deprecated. Instead, use the ISO C++ conformant name: _stricmp.
#define stricmp _stricmp
#endif

namespace VideoStitch {
namespace Output {

#define DEFAULT_I_QFACTOR -0.8f
#define DEFAULT_B_QFACTOR 1.25f
#define DEFAULT_I_QOFFSET 0.f
#define DEFAULT_B_QOFFSET 1.25f

typedef struct _EncodeConfig {
  int width;
  int height;
  int maxWidth;
  int maxHeight;
  FrameRate fps;
  int bitrate;
  int vbvMaxBitrate;
  int vbvSize;
  int rcMode;
  int qp;
  float i_quant_factor;
  float b_quant_factor;
  float i_quant_offset;
  float b_quant_offset;
  GUID presetGUID;
  GUID profileGUID;
  FILE* fOutput;
  int codec;
  int invalidateRefFramesEnableFlag;
  int intraRefreshEnableFlag;
  int intraRefreshPeriod;
  int intraRefreshDuration;
  int deviceType;
  int startFrameIdx;
  int endFrameIdx;
  int gopLength;
  int numB;
  int pictureStruct;
  int deviceID;
  int isYuv444;
  char* qpDeltaMapFile;

  std::string encoderPreset;
  std::string encoderProfile;

} EncodeConfig;

typedef struct _EncodeInputBuffer {
  unsigned int dwWidth;
  unsigned int dwHeight;
  void* pNV12devPtr;
  uint32_t uNV12Stride;
  void* nvRegisteredResource;
  NV_ENC_INPUT_PTR hInputSurface;
  NV_ENC_BUFFER_FORMAT bufferFmt;
} EncodeInputBuffer;

typedef struct _EncodeOutputBuffer {
  unsigned int dwBitstreamBufferSize;
  NV_ENC_OUTPUT_PTR hBitstreamBuffer;
  HANDLE hOutputEvent;
  bool bWaitOnEvent;
  bool bEOSFlag;
} EncodeOutputBuffer;

typedef struct _EncodeBuffer {
  EncodeOutputBuffer stOutputBfr;
  EncodeInputBuffer stInputBfr;
} EncodeBuffer;

typedef struct _NvEncPictureCommand {
  bool bResolutionChangePending;
  bool bBitrateChangePending;
  bool bForceIDR;
  bool bForceIntraRefresh;
  bool bInvalidateRefFrames;

  uint32_t newWidth;
  uint32_t newHeight;

  uint32_t newBitrate;
  uint32_t newVBVSize;

  uint32_t intraRefreshDuration;

  uint32_t numRefFramesToInvalidate;
  uint32_t refFrameNumbers[16];
} NvEncPictureCommand;

enum {
  NV_ENC_H264 = 0,
  NV_ENC_HEVC = 1,
};

template <class T>
class CNvQueue {
  T** m_pBuffer;
  unsigned int m_uSize;
  unsigned int m_uPendingCount;
  unsigned int m_uAvailableIdx;
  unsigned int m_uPendingndex;

 public:
  CNvQueue() : m_pBuffer(NULL), m_uSize(0), m_uPendingCount(0), m_uAvailableIdx(0), m_uPendingndex(0) {}

  ~CNvQueue() { delete[] m_pBuffer; }

  bool Initialize(T* pItems, unsigned int uSize) {
    m_uSize = uSize;
    m_uPendingCount = 0;
    m_uAvailableIdx = 0;
    m_uPendingndex = 0;
    m_pBuffer = new T*[m_uSize];
    for (unsigned int i = 0; i < m_uSize; i++) {
      m_pBuffer[i] = &pItems[i];
    }
    return true;
  }

  T* GetAvailable() {
    T* pItem = NULL;
    if (m_uPendingCount == m_uSize) {
      return NULL;
    }
    pItem = m_pBuffer[m_uAvailableIdx];
    m_uAvailableIdx = (m_uAvailableIdx + 1) % m_uSize;
    m_uPendingCount += 1;
    return pItem;
  }

  T* GetPending() {
    if (m_uPendingCount == 0) {
      return NULL;
    }

    T* pItem = m_pBuffer[m_uPendingndex];
    m_uPendingndex = (m_uPendingndex + 1) % m_uSize;
    m_uPendingCount -= 1;
    return pItem;
  }
};

/**
 * The NvEncoder is based on NvEnc, and supported on all
 * NVIDIA cards starting from the Kepler architecture.
 */

class NvEncoder : public VideoEncoder {
 public:
  NvEncoder(EncodeConfig&);
  ~NvEncoder();

  static Potential<VideoEncoder> createNvEncoder(const Ptv::Value& config, int width, int height, FrameRate framerate);
  static void supportedEncoders(std::vector<std::string>& codecs);

  bool encode(const Frame&, std::vector<VideoStitch::IO::DataPacket>&);
  void getHeaders(VideoStitch::IO::DataPacket& packet);

  char* metadata(char* enc, char* pend);
  int getBitRate() const { return m_stEncodeConfig.rcParams.averageBitRate / 1000; }
  bool dynamicBitrateSupported() const { return true; }
  bool setBitRate(uint32_t /*maxBitrate*/, uint32_t /*bufferSize*/);

 private:
  Status Initialize();
  NVENCSTATUS InitCuda(unsigned int);
  NVENCSTATUS AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight);
  NVENCSTATUS ReleaseIOBuffers();
  NVENCSTATUS ProcessOutput(const EncodeBuffer* pEncodeBuffer, std::vector<VideoStitch::IO::DataPacket>&);
  void ParseH264Bitstream(NV_ENC_LOCK_BITSTREAM&, mtime_t&, std::vector<VideoStitch::IO::DataPacket>&);
  void ParseHEVCBitstream(NV_ENC_LOCK_BITSTREAM&, mtime_t&, std::vector<VideoStitch::IO::DataPacket>&);

  NVENCSTATUS NvEncOpenEncodeSession(void* device, uint32_t deviceType);
  NVENCSTATUS NvEncGetEncodeGUIDCount(uint32_t* encodeGUIDCount);
  NVENCSTATUS NvEncGetEncodeProfileGUIDCount(GUID encodeGUID, uint32_t* encodeProfileGUIDCount);
  NVENCSTATUS NvEncGetEncodeProfileGUIDs(GUID encodeGUID, GUID* profileGUIDs, uint32_t guidArraySize,
                                         uint32_t* GUIDCount);
  NVENCSTATUS NvEncGetEncodeGUIDs(GUID* GUIDs, uint32_t guidArraySize, uint32_t* GUIDCount);
  NVENCSTATUS NvEncGetInputFormatCount(GUID encodeGUID, uint32_t* inputFmtCount);
  NVENCSTATUS NvEncGetInputFormats(GUID encodeGUID, NV_ENC_BUFFER_FORMAT* inputFmts, uint32_t inputFmtArraySize,
                                   uint32_t* inputFmtCount);
  NVENCSTATUS NvEncGetEncodeCaps(GUID encodeGUID, NV_ENC_CAPS_PARAM* capsParam, int* capsVal);
  NVENCSTATUS NvEncGetEncodePresetCount(GUID encodeGUID, uint32_t* encodePresetGUIDCount);
  NVENCSTATUS NvEncGetEncodePresetGUIDs(GUID encodeGUID, GUID* presetGUIDs, uint32_t guidArraySize,
                                        uint32_t* encodePresetGUIDCount);
  NVENCSTATUS NvEncGetEncodePresetConfig(GUID encodeGUID, GUID presetGUID, NV_ENC_PRESET_CONFIG* presetConfig);
  NVENCSTATUS NvEncCreateInputBuffer(uint32_t width, uint32_t height, void** inputBuffer, uint32_t isYuv444);
  NVENCSTATUS NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR inputBuffer);
  NVENCSTATUS NvEncCreateBitstreamBuffer(uint32_t size, void** bitstreamBuffer);
  NVENCSTATUS NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer);
  NVENCSTATUS NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM* lockBitstreamBufferParams);
  NVENCSTATUS NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer);
  NVENCSTATUS NvEncLockInputBuffer(void* inputBuffer, void** bufferDataPtr, uint32_t* pitch);
  NVENCSTATUS NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer);
  NVENCSTATUS NvEncGetEncodeStats(NV_ENC_STAT* encodeStats);
  NVENCSTATUS NvEncGetSequenceParams(NV_ENC_SEQUENCE_PARAM_PAYLOAD* sequenceParamPayload);
  NVENCSTATUS NvEncRegisterAsyncEvent(void** completionEvent);
  NVENCSTATUS NvEncUnregisterAsyncEvent(void* completionEvent);
  NVENCSTATUS NvEncMapInputResource(void* registeredResource, void** mappedResource);
  NVENCSTATUS NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer);
  NVENCSTATUS NvEncDestroyEncoder();
  NVENCSTATUS NvEncInvalidateRefFrames(const NvEncPictureCommand* pEncPicCommand);
  NVENCSTATUS NvEncOpenEncodeSessionEx(void* device, NV_ENC_DEVICE_TYPE deviceType);
  NVENCSTATUS NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister, uint32_t width,
                                    uint32_t height, uint32_t pitch, void** registeredResource);
  NVENCSTATUS NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes);
  NVENCSTATUS NvEncReconfigureEncoder(NV_ENC_RECONFIGURE_PARAMS* reInitEncodeParams);
  NVENCSTATUS NvEncFlushEncoderQueue(void* hEOSEvent);

  NVENCSTATUS InitializeAPI();
  NVENCSTATUS NvEncEncodeFrame(EncodeBuffer* pEncodeBuffer, NvEncPictureCommand* encPicCommand, uint32_t width,
                               uint32_t height, mtime_t pts, NV_ENC_PIC_STRUCT ePicStruct = NV_ENC_PIC_STRUCT_FRAME,
                               int8_t* qpDeltaMapArray = NULL, uint32_t qpDeltaMapArraySize = 0);
  NVENCSTATUS CreateEncoder(const EncodeConfig* pEncCfg);
  GUID GetPresetGUID(const char* encoderPreset, int codec);
  GUID GetProfileGUID(const char* encoderProfile, int codec);
  NVENCSTATUS FlushEncoder();
  NVENCSTATUS ValidateEncodeGUID(GUID inputCodecGuid);
  NVENCSTATUS ValidatePresetGUID(GUID presetCodecGuid, GUID inputCodecGuid);
  NVENCSTATUS ValidateProfileGUID(GUID profileCodecGuid, GUID inputCodecGuid);

  static const AVal av_videocodecid;
  static const AVal av_videodatarate;
  static const AVal av_framerate;

  EncodeConfig encodeConfig;

  HINSTANCE m_hinstLib;
  NV_ENCODE_API_FUNCTION_LIST* m_pEncodeAPI;
  void* m_hEncoder;
  bool m_bEncoderInitialized;
  NV_ENC_INITIALIZE_PARAMS m_stCreateEncodeParams;
  NV_ENC_CONFIG m_stEncodeConfig;
  GUID codecGUID;

  uint32_t m_uEncodeBufferCount;
  CNvQueue<EncodeBuffer> m_EncodeBufferQueue;
  EncodeBuffer m_stEncodeBuffer[MAX_ENCODE_QUEUE];
  EncodeOutputBuffer m_stEOSOutputBfr;

  std::queue<mtime_t> timestamps;
  int first_dts;
  FILE* m_fOutput;
  cudaStream_t stream;
};

typedef NVENCSTATUS(NVENCAPI* MYPROC)(NV_ENCODE_API_FUNCTION_LIST*);

}  // namespace Output
}  // namespace VideoStitch
