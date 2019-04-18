// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/frame.hpp"

namespace VideoStitch {
namespace Testing {

void testFrameRateEquality() {
  ENSURE(FrameRate() == FrameRate());
  ENSURE(FrameRate(-1, -1) == FrameRate(-1, -1));

  ENSURE(FrameRate(30, 1) == FrameRate(30, 1));
  ENSURE(FrameRate(30, 1) != FrameRate(25, 1));

  ENSURE(FrameRate(30000, 1000) == FrameRate(30, 1));
  ENSURE(FrameRate(30, 1) == FrameRate(30000, 1000));

  ENSURE(FrameRate(30000, 1001) == FrameRate(30000, 1001));
  ENSURE(FrameRate(30000, 1001) != FrameRate(30000, 1000));

  ENSURE(FrameRate(0, 0) != FrameRate(25, 1));
  ENSURE(FrameRate(25, 1) != FrameRate(0, 0));

  ENSURE(FrameRate(0, 0) == FrameRate(0, 0));

  ENSURE(FrameRate(25, 0) == FrameRate(25, 0));
  ENSURE(FrameRate(25, 0) != FrameRate(30, 0));
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testFrameRateEquality();
  return 0;
}
