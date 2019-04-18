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
 * Procedural Checkerboard reader.
 */
class CheckerBoardReader : public VideoReader {
 public:
  CheckerBoardReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth, int64_t targetHeight);
  virtual ~CheckerBoardReader();

  ReadStatus readFrame(mtime_t& date, unsigned char* video);
  Status seekFrame(frameid_t);

 private:
  int checkerSize;
  uint32_t color1;
  uint32_t color2;
  uint32_t color3;
  mtime_t curDate;
};
}  // namespace Input
}  // namespace VideoStitch
