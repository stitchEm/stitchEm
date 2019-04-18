// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "libvideostitch/config.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace Util {

class VS_EXPORT Compression {
 public:
  // https ://developers.google.com/maps/documentation/utilities/polylinealgorithm
  // https://gist.github.com/shinyzhu/4617989
  static Status polylineEncodeBinaryMask(const int width, const int height, const std::vector<unsigned char>& data,
                                         std::string& values);
  static Status polylineDecodeBinaryMask(const int width, const int height, const std::string& values,
                                         std::vector<unsigned char>& data, const bool toBinary = true);

  static Status polylineEncodeBinaryMask(const int width, const int height, const std::string& data,
                                         std::string& values);
  static Status polylineDecodeBinaryMask(const int width, const int height, const std::string& values,
                                         std::string& data);

  static float binaryDifference(const std::vector<unsigned char>& im0, const std::vector<unsigned char>& im1);
  // Convert mask <--> encoded masks
  static Status convertMaskToEncodedMasks(const int width, const int height, const uint32_t* const fullData,
                                          std::vector<std::string>& datas);
  static Status convertEncodedMasksToMask(const int width, const int height, const std::vector<std::string>& datas,
                                          std::vector<uint32_t>& fullData);
};

}  // namespace Util
}  // namespace VideoStitch
