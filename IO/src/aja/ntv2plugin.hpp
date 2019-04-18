// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ajatypes.h"
#include "ajastuff/common/types.h"
#include "ntv2card.h"
#include "ntv2publicinterface.h"
#include "ntv2rp188.h"

#include "libvideostitch/frame.hpp"

#include "ntv2Helper.hpp"

#include <map>
#include <mutex>

static const ULWord appSignature AJA_FOURCC('V', 'I', 'S', 'T');

namespace VideoStitch {

/**
 This structure encapsulates the video and audio buffers.
 The producer/consumer threads use a fixed number (CIRCULAR_BUFFER_SIZE) of these buffers.
 The AJACircularBuffer template class greatly simplifies implementing this approach to efficiently
 process frames.
**/
typedef struct {
  uint32_t* videoBuffer;          ///	Pointer to host video buffer
  uint32_t* videoBuffer2;         ///	Pointer to an additional host video buffer, usually field 2
  uint32_t videoBufferSize;       ///	Size of host video buffer, in bytes
  uint32_t inNumSegments;         /// 1 for host video buffer transfer, number of lines for specialized data transfers
  uint32_t inDeviceBytesPerLine;  /// device pitch for specialized data transfers
  uint32_t* audioBuffer;          ///	Pointer to host audio buffer
  uint32_t audioBufferSize;       ///	Size of host audio buffer, in bytes
  CRP188 rp188;                   ///  Time and control code
  uint32_t* ancBuffer;
  uint32_t ancBufferSize;
  uint32_t currentFrame;    ///    Frame Number
  uint64_t audioTimeStamp;  ///    Audio TimeStamp
} AVDataBuffer;

const unsigned int CIRCULAR_BUFFER_SIZE(10);  ///	Specifies how many AVDataBuffers constitute the circular buffer

class NTV2Device {
 public:
  static NTV2Device* getDevice(uint32_t device);
  virtual ~NTV2Device();

  CNTV2Card device;
  NTV2DeviceID deviceID;

 private:
  NTV2Device();
  AJAStatus init(uint32_t deviceIndex);

  NTV2EveryFrameTaskMode savedTaskMode;  /// Used to restore prior every-frame task mode
  bool initialized;
};

static std::mutex registryMutex;
static std::map<int, NTV2Device*> registry;

}  // namespace VideoStitch
