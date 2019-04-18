// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PNGREADER_HPP_
#define PNGREADER_HPP_

#include <string>
#include <stdint.h>
#include <vector>
#ifndef PNGLIB_UNSUPPORTED

#ifdef _MSC_VER
#include <png.h>
#else
#include <libpng16/png.h>
#endif

#endif

namespace VideoStitch {
namespace Util {

/**
 * A class to read PNGs.
 * Not thread safe.
 */
class PngReader {
 public:
  PngReader();

  /**
   * Read a PNG with the given size.
   * @param filename file to read from.
   * @param width Expected image width.
   * @param height Expected image height.
   * @param data buffer to hold the output. Should be of size width * height * 4.
   * @return true on success.
   */
  static bool readRGBAFromFile(const char* filename, int64_t width, int64_t height, void* data);

  /**
   * Read a PNG with the given size.
   * @param filename file to read from.
   * @param width The image width.
   * @param height The image height.
   * @param data Vector to hold the output. Should be of size width * height * 4.
   * @return true on success.
   */
  static bool readRGBAFromFile(const char* filename, int64_t& width, int64_t& height, std::vector<unsigned char>& data);

  /**
   * Read a monochrome PNG with the given size.
   * @param filename file to read from.
   * @param width Expected image width.
   * @param height Expected image height.
   * @param data buffer to hold the output. Should be of size width * height.
   * @return true on success.
   */
  static bool readMonochromeFromFile(const char* filename, int64_t width, int64_t height, void* data);

  /**
   * Read a mask (colormapped) PNG with the given size from memory.
   * @param input data to read from.
   * @param inputLen Size of @a data.
   * @param width On output, contains the image width.
   * @param height On output, contains the image height.
   * @param data buffer that will be malloc()ed to hold the output.
   * @return true on success.
   */
  static bool readMaskFromMemory(const unsigned char* input, size_t inputLen, int64_t& width, int64_t& height,
                                 void** data);

  /**
   * Write a mask (2- color colormapped) PNG with the given size to memory. Only values are 0 and 1.
   * @param output Output buffer.
   * @param width Expected image width.
   * @param height Expected image height.
   * @param data buffer that holds the input. Should be of size width * height.
   * @return true on success.
   */
  static bool writeMaskToMemory(std::string& output, int64_t width, int64_t height, const void* data);

  /**
   * Write a PNG with the given size.
   * @param filename file to write to.
   * @param width Image width.
   * @param height Image height.
   * @param data RGBA buffer that holds the output. Should be of size width * height * 4.
   * @return true on success.
   */
  static bool writeRGBAToFile(const char* filename, int64_t width, int64_t height, const void* data);

  /**
   * @brief Write a PNG given the size (intended to work with a cv::Mat CV_8UC3)
   * @param filename: output filename
   * @param data: BGR buffer that holds the input. Should be of size width * height 3 bytes channels continuous
   * @return true on success
   */
  static bool writeBGRToFile(const char* filename, int64_t width, int64_t height, const void* data);

  /**
   * Write a PNG of mono (8UC1)
   * @param filename file to write to.
   * @param width Image width.
   * @param height Image height.
   * @param data buffer that holds the output. Should be of size width * height.
   * @return true on success.
   */
  static bool writeMonochromToFile(const char* filename, int64_t width, int64_t height, const void* data);

  /**
   * Write a PNG of mono (16UC1)
   * @param filename file to write to.
   * @param width Image width.
   * @param height Image height.
   * @param data buffer that holds the output. Should be of size width * height.
   * @return true on success.
   */
  static bool writeMonochrome16ToFile(const char* filename, int64_t width, int64_t height, const void* data);

 private:
#ifndef PNGLIB_UNSUPPORTED
  // TODO: remove ifdef, Waiting for libs...
  static png_image image;
#endif
};

}  // namespace Util
}  // namespace VideoStitch

#endif
