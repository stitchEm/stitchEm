// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"

namespace VideoStitch {
namespace Input {

/**
 * Procedural Profiling reader.
 */
class ProfilingReader : public VideoReader {
 public:
  ProfilingReader(readerid_t id, int64_t targetWidth, int64_t targetHeight);
  virtual ~ProfilingReader();

  ReadStatus readFrame(mtime_t& date, unsigned char* video);
  Status seekFrame(frameid_t);

 private:
  mtime_t curDate;
  unsigned char* inputFrame;
};
}  // namespace Input
}  // namespace VideoStitch
