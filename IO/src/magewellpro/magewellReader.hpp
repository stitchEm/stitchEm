// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/ptv.hpp"

#include <windows.h>

#include "LibMWCapture/MWCapture.h"

#include <vector>
#include <condition_variable>
#include <mutex>

/**
 * Magewell capture card adapter.
 */
namespace VideoStitch {
namespace Input {

class MagewellReader : public VideoReader {
 public:
  static MagewellReader* create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height);

  virtual ~MagewellReader();

  virtual ReadStatus readFrame(mtime_t& frame, unsigned char* video);
  virtual Status seekFrame(frameid_t date);

 private:
  MagewellReader(readerid_t id, const int64_t width, const int64_t height, HCHANNEL channel, int64_t frameSize,
                 FrameRate fps, PixelFormat pixelFormat, int bytesPerPixel, const std::string& name);
  bool init();

  HCHANNEL channel;
  const std::string name;
  const int bytesPerPixel;
  DWORD format;
  HANDLE notifyEvent, captureEvent;
  HNOTIFY notify;
  MWCAP_VIDEO_BUFFER_INFO videoBufferInfo;
  MWCAP_VIDEO_FRAME_INFO videoFrameInfo;
  MWCAP_VIDEO_SIGNAL_STATUS videoSignalStatus;

  LONGLONG totalTime = 0LL;
  UINT64 timestamp;
};

}  // namespace Input
}  // namespace VideoStitch
