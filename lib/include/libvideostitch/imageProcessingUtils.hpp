// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "config.hpp"
#include "status.hpp"
#include "ptv.hpp"

namespace VideoStitch {
namespace Util {

class VS_EXPORT ImageProcessing {
 public:
  static Status findCropCircle(const int& width, const int& height, void* data, int& x, int& y, int& r,
                               const VideoStitch::Ptv::Value* algoConfig = nullptr,
                               const std::string* dumpFile = nullptr);
  /**
   * @brief Read an image file
   * @IMPORTANT: Only support PNG for now
   */
  static Status readImage(const std::string& filename, int64_t& width, int64_t& height, int& channelCount,
                          std::vector<unsigned char>& data);

  static Status packImageRGBA(const std::vector<unsigned char>& data, std::vector<uint32_t>& rgba);
  static void unpackImageRGBA(const std::vector<uint32_t>& rgba, std::vector<unsigned char>& data);

  template <class T>
  static void convertIndexToRGBA(const std::vector<T>& hostRGBA, std::vector<unsigned char>& data,
                                 const int displayBit = -1);

  static uint32_t packRGBAColor(const uint32_t r, const uint32_t g, const uint32_t b, const uint32_t a);
  static uint32_t packRGB210Color(const uint32_t r, const uint32_t g, const uint32_t b, const uint32_t a);
};

}  // namespace Util
}  // namespace VideoStitch
