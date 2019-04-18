// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <string>
#include <stdint.h>

#include <libvideostitch/status.hpp>

#ifdef _MSC_VER
#include <png.h>
#else
#include <libpng16/png.h>
#endif

namespace VideoStitch {
namespace detail {

/**
 * A class to read PNGs.
 * Not thread safe.
 */
class Png {
 public:
  /**
   * Read a mask (colormapped) PNG with the given size from memory.
   * @param filename file to read from.
   * @param width On output, contains the image width.
   * @param height On output, contains the image height.
   * @return true on success.
   */
  Status readHeader(const char* filename, int64_t& width, int64_t& height);

  /**
   * Read a PNG with the given size.
   * @param filename file to read from.
   * @param width Expected image width.
   * @param height Expected image height.
   * @param data buffer to hold the output. Should be of size width * height * 4.
   * @return true on success.
   */
  Status readRGBAFromFile(const char* filename, int64_t width, int64_t height, void* data);

  /**
   * Write a PNG with the given size.
   * @param filename file to write to.
   * @param width Image width.
   * @param height Image height.
   * @param data RGB buffer that holds the output. Should be of size width * height * 3.
   */
  Status writeRGBToFile(const char* filename, int64_t width, int64_t height, const void* data);

  /**
   * Write a PNG with the given size.
   * @param filename file to write to.
   * @param width Image width.
   * @param height Image height.
   * @param data RGBA buffer that holds the output. Should be of size width * height * 4.
   */
  Status writeRGBAToFile(const char* filename, int64_t width, int64_t height, const void* data);

  /**
   * Write a PNG of mono (16UC1)
   * @param filename file to write to.
   * @param width Image width.
   * @param height Image height.
   * @param data buffer that holds the output. Should be of size width * height.
   */
  Status writeMonochrome16ToFile(const char* filename, int64_t width, int64_t height, const void* data);

 private:
  png_image image;
};
}  // namespace detail
}  // namespace VideoStitch
