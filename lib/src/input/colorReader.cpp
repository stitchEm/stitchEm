// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "colorReader.hpp"

#include "backend/common/imageOps.hpp"

#include "gpu/stream.hpp"

#include "gpu/render/render.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <cassert>
#include <iostream>
#include <limits>

#define DEFAULT_COLOR Image::RGBA::pack(0xff, 0x00, 0x00, 0xff)

namespace VideoStitch {
namespace Input {

ColorReader::ColorReader(readerid_t id, const Ptv::Value& config, int64_t targetWidth, int64_t targetHeight)
    : Reader(id),
      VideoReader(targetWidth, targetHeight, targetWidth * targetHeight * sizeof(uint32_t), RGBA, Device,
                  {60, 1} /*fps*/, 0, NO_LAST_FRAME, true /* procedural */, NULL),
      fillColor(DEFAULT_COLOR),
      curDate(-1) {
  Parse::populateColor("ReaderConfig", config, "color", fillColor, false);
  getSpec().setDisplayName("Procedural: Color");
}

ColorReader::~ColorReader() {}

Status ColorReader::seekFrame(frameid_t) { return Status::OK(); }

ReadStatus ColorReader::readFrame(mtime_t& date, unsigned char* videoFrame) {
  // Everything is done on the GPU
  // XXX TODO FIXME procedurals with a frame rate please
  curDate += (mtime_t)round(getSpec().frameRate.den / (double)getSpec().frameRate.num * 1000000.0);
  date = curDate;
  Render::fillBuffer(GPU::Buffer<uint32_t>::wrap((uint32_t*)videoFrame, getWidth() * getHeight()), fillColor,
                     getWidth(), getHeight(), GPU::Stream::getDefault());
  return ReadStatus::OK();
}
}  // namespace Input
}  // namespace VideoStitch
