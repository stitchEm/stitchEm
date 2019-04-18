// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "libvideostitch/config.hpp"

#include <opencv2/imgproc.hpp>

#include <map>

namespace VideoStitch {
namespace Util {

class VS_EXPORT PolylineEncoding {
 public:
  // https ://developers.google.com/maps/documentation/utilities/polylinealgorithm
  // https://gist.github.com/shinyzhu/4617989

  static void polylineEncode(const std::vector<cv::Point2f>& polyPoints, std::string& encoded,
                             const float scale = 1e5f);
  static void polylineDecode(const std::string& encoded, std::vector<cv::Point2f>& polyPoints,
                             const float scale = 1e5f);

  static void polylineEncode(const std::vector<cv::Point>& polyPoints, std::string& encoded);
  static void polylineDecode(const std::string& encoded, std::vector<cv::Point>& polyPoints);

  static void polylineDecodePolygon(const std::string& values, std::vector<std::vector<cv::Point>>& points);
  static void polylineDecodePolygons(const std::map<videoreaderid_t, std::string>& encodedValues,
                                     std::map<videoreaderid_t, std::vector<cv::Point>>& decodedPolygons);

  static std::string polylineEncodeValue(const int32_t value);
  static int32_t polylineDecodeValue(const std::string& encoded, int32_t& index, int32_t& next5bits);
};

}  // namespace Util
}  // namespace VideoStitch
