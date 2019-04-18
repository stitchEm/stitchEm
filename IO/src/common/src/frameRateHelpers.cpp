// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <cmath>
#include "frameRateHelpers.hpp"

namespace VideoStitch {
namespace Util {

FrameRate fpsToNumDen(const double fps) {
  FrameRate frameRate;
  if (fabs(fps - (int)fps) < 0.001) {
    // infer integer frame rates
    frameRate.num = (int)fps;
    frameRate.den = 1;
  } else if (fabs((fps * 1001.0) / 1000.0 - (int)(fps + 1)) < 0.001) {
    // infer ATSC frame rates
    frameRate.num = (int)(fps + 1) * 1000;
    frameRate.den = 1001;
  } else if (fabs(fps * 2 - (int)(fps * 2)) < 0.001) {
    // infer rational frame rates; den = 2
    frameRate.num = (int)(fps * 2);
    frameRate.den = 2;
  } else if (fabs(fps * 4 - (int)(fps * 4)) < 0.001) {
    // infer rational frame rates; den = 4
    frameRate.num = (int)(fps * 4);
    frameRate.den = 4;
  } else {
    frameRate.num = (int)fps;
    frameRate.den = 1;
  }
  return frameRate;
}

}  // namespace Util
}  // namespace VideoStitch
