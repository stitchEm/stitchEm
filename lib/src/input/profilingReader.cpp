// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "profilingReader.hpp"

#include "libvideostitch/profile.hpp"
#include "libvideostitch/logging.hpp"

#include <cmath>

namespace VideoStitch {
namespace Input {

ProfilingReader::ProfilingReader(readerid_t id, int64_t targetWidth, int64_t targetHeight)
    : Reader(id),
      VideoReader(targetWidth, targetHeight, (targetWidth * targetHeight * 3) / 2, YV12, Host, {60, 1} /*fps*/, 0,
                  NO_LAST_FRAME, true /* procedural */, NULL),
      curDate(0) {
  inputFrame = (unsigned char*)malloc(getSpec().frameDataSize);
  for (int i = 0; i < getSpec().frameDataSize; i++) {
    inputFrame[i] = (unsigned char)rand();
  }
  getSpec().setDisplayName("Procedural: Profiling");
}

ProfilingReader::~ProfilingReader() { free(inputFrame); }

Status ProfilingReader::seekFrame(frameid_t) { return Status::OK(); }

ReadStatus ProfilingReader::readFrame(mtime_t& date, unsigned char* videoFrame) {
  // XXX TODO FIXME procedurals with a frame rate please
  curDate += (mtime_t)round(getSpec().frameRate.den / (double)getSpec().frameRate.num * 1000000.0);
  date = curDate;

  // can't simulate realistic usage, but at the very least the decoder has to write all the data once
  memcpy(videoFrame, inputFrame, getSpec().frameDataSize);
  return ReadStatus::OK();
}

}  // namespace Input
}  // namespace VideoStitch
