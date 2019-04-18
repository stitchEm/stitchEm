// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "checkerBoardReader.hpp"

#include "backend/common/imageOps.hpp"

#include "gpu/stream.hpp"

#include "libvideostitch/parse.hpp"
#include "gpu/input/checkerBoard.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <sstream>

#define DEFAULT_SIZE 32
#define DEFAULT_COLOR1 Image::RGBA::pack(0x00, 0x00, 0x00, 0xff)
#define DEFAULT_COLOR2 Image::RGBA::pack(0xff, 0xff, 0xff, 0xff)
#define DEFAULT_COLOR3 Image::RGBA::pack(0x22, 0x22, 0x22, 0xff)

namespace VideoStitch {
namespace Input {

CheckerBoardReader::CheckerBoardReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth,
                                       int64_t targetHeight)
    : Reader(id),
      VideoReader(targetWidth, targetHeight, targetWidth * targetHeight * sizeof(uint32_t), RGBA, Device,
                  {60, 1} /*fps*/, 0, NO_LAST_FRAME, true /* procedural */, NULL),
      checkerSize(DEFAULT_SIZE),
      color1(DEFAULT_COLOR1),
      color2(DEFAULT_COLOR2),
      color3(DEFAULT_COLOR3),
      curDate(0) {
  Parse::populateInt("ReaderConfig", config, "size", checkerSize, false);
  Parse::populateColor("ReaderConfig", config, "color1", color1, false);
  Parse::populateColor("ReaderConfig", config, "color2", color2, false);
  Parse::populateColor("ReaderConfig", config, "color3", color3, false);
  getSpec().setDisplayName("Procedural: CheckerBoard");
}

CheckerBoardReader::~CheckerBoardReader() {}

Status CheckerBoardReader::seekFrame(frameid_t) { return Status::OK(); }

ReadStatus CheckerBoardReader::readFrame(mtime_t& date, unsigned char* videoFrame) {
  // Everything is done on the GPU
  // XXX TODO FIXME procedurals with a frame rate please
  curDate += (mtime_t)round(getSpec().frameRate.den / (double)getSpec().frameRate.num * 1000000.0);
  date = curDate;
  overlayCheckerBoard(GPU::Buffer<uint32_t>::wrap((uint32_t*)videoFrame, getWidth() * getHeight()),
                      (unsigned)getWidth(), (unsigned)getHeight(), checkerSize, color1, color2, color3,
                      GPU::Stream::getDefault());
  return ReadStatus::OK();
}
}  // namespace Input
}  // namespace VideoStitch
