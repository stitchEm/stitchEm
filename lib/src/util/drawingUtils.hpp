// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"
#include <opencv2/imgproc.hpp>

namespace VideoStitch {
namespace Util {

class VS_EXPORT Drawing {
 public:
  static void line(cv::Mat& img, const cv::Point& start, const cv::Point& end, const cv::Scalar& c1,
                   const cv::Scalar& c2);
  static void polylines(cv::Mat& img, const std::vector<cv::Point>& points, const cv::Scalar& c1, const cv::Scalar& c2);
};

}  // namespace Util
}  // namespace VideoStitch
