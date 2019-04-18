// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <stdint.h>

namespace VideoStitch {
namespace Util {

/**
 * A class to read EXR files.
 */
class Exr {
 public:
  /**
   * Write a floating-point single-channel depth image
   * @param filename file to write to.
   * @param width Image width.
   * @param height Image height.
   * @param data buffer that holds the output. Should be of size width * height.
   * @return true on success.
   */
  static bool writeDepthToFile(const char* filename, int64_t width, int64_t height, const float* data);
};

}  // namespace Util
}  // namespace VideoStitch
