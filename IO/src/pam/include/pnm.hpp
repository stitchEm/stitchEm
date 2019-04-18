// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/logging.hpp"

#include <fstream>
#include <ostream>
#include <vector>
#include <stdint.h>

namespace VideoStitch {
namespace Util {
/**
 * @brief A class that reads PNM images (PBM, PGM, PPM).
 */
class PnmReader {
 public:
  /**
   * Reads an image from @a filename.
   * @param filename Input file name.
   * @param w On return, contains the width of the image.
   * @param h On return, contains the height of the image.
   * @param data On return, contains the data for the image, in RGB format (RGBA is @a pad is true).
   * @param err If not NULL, error messages are written there.
   * @param pad If true, @data will be RGBA.
   */
  static bool read(const char *filename, int64_t &w, int64_t &h, std::vector<unsigned char> &data,
                   std::ostream *err = NULL, bool pad = false);

  /**
   * Reads an image from an already open file.
   * @param ifs Input stream.
   * @param w On return, contains the width of the image.
   * @param h On return, contains the height of the image.
   * @param data On return, contains the data for the image, in RGB format (RGBA is @a pad is true).
   * @param err If not NULL, error messages are written there.
   * @param pad If true, @data will be RGBA.
   */
  static bool read(std::ifstream &ifs, int64_t &w, int64_t &h, std::vector<unsigned char> &data,
                   std::ostream *err = NULL, bool pad = false);

 private:
  enum PixType { AsciiPBM, AsciiPGM, AsciiPPM, BinPBM, BinPGM, BinPPM };

  static bool _readCommentWidthHeight(std::ifstream &ifs, int64_t &width, int64_t &height, std::ostream *err);

  // binary RGB
  template <PixType type>
  static bool _read(std::ifstream &ifs, int64_t &w, int64_t &h, std::vector<unsigned char> &data, bool pad,
                    std::ostream *err);

  template <PixType type>
  static void _readPixel(std::ifstream &ifs, std::vector<unsigned char> &data);

  static const size_t bufSize = 512;
};

/**
 * @brief PNM writer class.
 */
class PpmWriter {
 public:
  /**
   * Open a Ppm file for writing and write the header.
   * @param filename Name of the file to open.
   * @param w Image width.
   * @param h image height.
   * @param err Output stream for errors.
   * @return NULL on error.
   */
  static std::ofstream *openPpm(const char *filename, int64_t w, int64_t h, VideoStitch::ThreadSafeOstream *err = NULL);

  /**
   * Open a Pam file for writing and write the header.
   * @param filename Name of the file to open.
   * @param w Image width.
   * @param h image height.
   * @param err Output stream for errors.
   * @return NULL on error.
   */
  static std::ofstream *openPam(const char *filename, int64_t w, int64_t h, VideoStitch::ThreadSafeOstream *err = NULL);

  /**
   * Open a Pgm file for writing and write the header.
   * @param filename Name of the file to open.
   * @param w Image width.
   * @param h image height.
   * @param err Output stream for errors.
   * @return NULL on error.
   */
  static std::ofstream *openPgm(const char *filename, int64_t w, int64_t h, VideoStitch::ThreadSafeOstream *err = NULL);

 private:
  static std::ofstream *openGeneric(const char *filename, VideoStitch::ThreadSafeOstream *err = NULL);
};
}  // namespace Util
}  // namespace VideoStitch
