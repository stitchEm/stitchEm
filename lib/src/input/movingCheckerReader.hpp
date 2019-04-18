// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"

namespace VideoStitch {
namespace Input {

/**
 * Procedural reader that creates moving checker boards on the host
 */
class MovingCheckerReader : public VideoReader {
 public:
  MovingCheckerReader(readerid_t id, int64_t targetWidth, int64_t targetHeight);
  virtual ~MovingCheckerReader();

  ReadStatus readFrame(mtime_t& date, unsigned char* video);
  Status seekFrame(frameid_t);

 private:
  frameid_t curFrame;
};
}  // namespace Input
}  // namespace VideoStitch
