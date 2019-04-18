// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/ptv.hpp"

#include <windows.h>
#include "LibXIProperty/XIProperty.h"
#include "LibXIStream/XIStream.h"

#include <deque>
#include <vector>
#include <unordered_map>
#include <condition_variable>
#include <mutex>

/**
 * Magewell capture card adapter.
 */
namespace VideoStitch {
namespace Input {

class MagewellReader : public VideoReader {
 public:
  virtual ~MagewellReader();

  static MagewellReader* create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height);
  static bool handles(const Ptv::Value* config);

  ReadStatus readFrame(mtime_t& date, unsigned char* video);
  Status seekFrame(frameid_t date);

 private:
  MagewellReader(readerid_t id, const int64_t width, const int64_t height, HANDLE videoCapture, int64_t frameSize,
                 FrameRate fps, const std::string& name);

  static void videoCallback(const BYTE* pbyImage, int cbImageStride, void* pvParam, UINT64 u64TimeStamp);

  // Helpers
  static VIDEO_CAPTURE_INFO_EX retrieveVideoCaptureInfo(const std::string& name, bool& ok);
  static XIPHD_SCALE_TYPE getScaleType(const Ptv::Value& config, bool& ok);
  static void getPixelFormatInfo(const Ptv::Value& config, PixelFormat& pixelFormat, int& bytesPerPixel, bool& ok);

  // Variables
  HANDLE videoCapture;
  const std::string name;

  std::mutex videoMu;
  std::condition_variable videoCv;
  bool frameAvailable;

  std::vector<unsigned char> videoFrame;
  UINT64 timestamp;
};

}  // namespace Input
}  // namespace VideoStitch
