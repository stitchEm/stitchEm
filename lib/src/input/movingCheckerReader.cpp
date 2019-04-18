// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "movingCheckerReader.hpp"

#include "libvideostitch/logging.hpp"

#include <cmath>

static const int CHECKER_SIZE = 32;
static const int MAX_CHROMA = 24;

namespace VideoStitch {
namespace Input {

MovingCheckerReader::MovingCheckerReader(readerid_t id, int64_t targetWidth, int64_t targetHeight)
    : Reader(id),
      VideoReader(targetWidth, targetHeight, (targetWidth * targetHeight * 3) / 2, VideoStitch::YV12, Host,
                  {60, 1} /*fps*/, 0, NO_LAST_FRAME, true /* procedural */, NULL),
      curFrame(0) {
  getSpec().setDisplayName("Procedural: MovingChecker");
}

MovingCheckerReader::~MovingCheckerReader() {}

void movingCheckerBoard(unsigned char* dst, int64_t width, int64_t height, int checkerOffset) {
  // checkerboard pattern in Y
  for (int64_t y = 0; y < height; y++) {
    unsigned evenCol = (((y + 2 * checkerOffset) / CHECKER_SIZE) & 1);
    for (int64_t x = 0; x < width; x++) {
      unsigned evenRow = (((x + 4 * checkerOffset) / CHECKER_SIZE) & 1);
      dst[y * width + x] = (evenRow ^ evenCol) ? 64 : 196;
    }
  }

  // slowly changing color scheme
  auto uval = (unsigned char)((sin(checkerOffset / 10. * M_PI)) * MAX_CHROMA + 128);
  auto vval = (unsigned char)((cos(checkerOffset / 51. * M_PI)) * MAX_CHROMA + 128);

  unsigned char* uv = &dst[width * height];
  // U
  for (int64_t y = 0; y < height / 4; y++) {
    for (int64_t x = 0; x < width; x++) {
      uv[y * width + x] = uval;
    }
  }

  unsigned char* v = &uv[width * height / 4];
  // V
  for (int64_t y = 0; y < height / 4; y++) {
    for (int64_t x = 0; x < width; x++) {
      v[y * width + x] = vval;
    }
  }
}

Status MovingCheckerReader::seekFrame(frameid_t frame) {
  curFrame = frame;
  return Status::OK();
}

ReadStatus MovingCheckerReader::readFrame(mtime_t& date, unsigned char* videoFrame) {
  // XXX TODO FIXME procedurals with a frame rate please
  date =
      (mtime_t)round(1000000.0 * (double)curFrame * (double)getSpec().frameRate.den / (double)getSpec().frameRate.num);

  // can't simulate realistic usage, but at the very least the decoder has to write all the data once
  // memcpy(videoFrame, inputFrame, getSpec().frameDataSize);
  movingCheckerBoard(videoFrame, getSpec().width, getSpec().height, curFrame);
  curFrame++;
  return ReadStatus::OK();
}

}  // namespace Input
}  // namespace VideoStitch
