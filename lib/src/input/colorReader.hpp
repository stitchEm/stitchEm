// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"
#include "libvideostitch/ptv.hpp"
#include "proceduralParser.hpp"

#include <string>

namespace VideoStitch {
namespace Input {
/**
 * A full-GPU uniform color reader.
 */
class ColorReader : public VideoReader {
 public:
  ColorReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth, int64_t targetHeight);
  virtual ~ColorReader();

  ReadStatus readFrame(mtime_t& date, unsigned char* videoFrame);
  Status seekFrame(frameid_t);

 private:
  uint32_t fillColor;
  mtime_t curDate;
};
}  // namespace Input
}  // namespace VideoStitch
