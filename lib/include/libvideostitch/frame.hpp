// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include <iostream>
#include <cmath>

namespace VideoStitch {

/**
 * Frame rate. http://en.wikipedia.org/wiki/Frame_rate
 * @brief Defines the frame rate at which the system is functioning.
 */
struct VS_EXPORT FrameRate {
  /** Double Million constant **/
  static const int MILLION = 1000000;
  /** Numerator */
  int num;
  /** Denominator */
  int den;

  /** default constructor, arguments are "unknown framerate" */
  FrameRate(int numerator = -1, int denominator = 1) : num(numerator), den(denominator) {}

  /** Equality operator
   *  True when framerates are valid and exactly equal or offset by an integer factor
   *  True when framerates are invalid and exactly equal
   */
  bool operator==(const FrameRate& rhs) const {
    if (num != 0 && den != 0 && rhs.num != 0 && rhs.den != 0) {
      return num * rhs.den == rhs.num * den;
    }
    return num == rhs.num && den == rhs.den;
  }

  /** Equality operator */
  bool operator!=(const FrameRate& rhs) const { return !(*this == rhs); }

  /**
   * @brief Helper method to convert frameid to timestamp
   * @param frame input frame id
   * @return timestamp
   */
  mtime_t frameToTimestamp(frameid_t frame) const {
    return (mtime_t)round((double)frame * (double)den * MILLION / (double)num);
  }

  /**
   * @brief Helper method to convert timestamp to frameid
   * @param timestamp input timestamp
   * @return frameid
   */
  frameid_t timestampToFrame(mtime_t timestamp) const {
    return (frameid_t)round(((double)timestamp * (double)num / MILLION / (double)den));
  }
};

inline std::ostream& operator<<(std::ostream& os, const FrameRate& frameRate) {
  return os << frameRate.num << "/" << frameRate.den;
}

/**
 * Pixel format. Also see: http://www.fourcc.org/fourcc.php
 * @brief Defines the supported pixel formats in VideoStitch.
 */
enum PixelFormat {
  // RGB
  RGBA,
  RGB,
  // BGR
  BGR,
  BGRU,
  // 4:2:2
  UYVY,       // 16bpp
  YUY2,       // 16bpp
  YUV422P10,  // 20bpp, planar YUV 4:2:2, (1 Cr & Cb per 2x1 Y samples)
  // 4:2:0
  YV12,  // 12bpp, planar
  NV12,  // 12bpp, UV interleaved
  // monochrome
  Grayscale,
  // monochrome 16bpp
  Grayscale16,
  // demosaicing
  Bayer_RGGB,
  Bayer_BGGR,
  Bayer_GRBG,
  Bayer_GBRG,
  F32_C1,
  // depth map encoding
  DEPTH,

  Unknown
};

/**
 * @brief Return a string representation of a pixel format.
 * This can be used when serializing/parsing a pixel format enum.
 * @param pixelFormat A video stitch pixel format
 * @return A string representing the pixel format
 */
VS_EXPORT const char* getStringFromPixelFormat(const PixelFormat pixelFormat);

/**
 * @brief Return a Pixel format from a string.
 * @param name Pixel format string name
 * @return A pixel format enum
 */
VS_EXPORT PixelFormat getPixelFormatFromString(const std::string& name);

/**
 * @brief Return a Frame buffer size in bytes according to Pixel format.
 * @param width A video width
 * @param height A video stitch pixel format
 * @param pixelFormat A video stitch pixel format
 * @return A Frame buffer size
 */
VS_EXPORT int32_t getFrameDataSize(const int32_t width, const int32_t height, const PixelFormat pixelFormat);

/**
 * Video frame.
 */
struct Frame {
  /**
   * Pointer to the picture planes.
   */
  void* planes[3];

  /**
   * Size in bytes of each picture line.
   */
  size_t pitches[3];

  /**
   * Width of the video frame in pixels.
   */
  int32_t width;

  /**
   * Height of the video frame in pixels.
   */
  int32_t height;

  /**
   * Presentation timestamp in microseconds of the video frame.
   */
  mtime_t pts;

  /**
   * Pixel format of the video frame.
   */
  PixelFormat fmt;
};

}  // namespace VideoStitch
