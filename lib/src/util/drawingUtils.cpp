// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "drawingUtils.hpp"
#include "pngutil.hpp"
#include <opencv2/core.hpp>

namespace VideoStitch {
namespace Util {

void Drawing::line(cv::Mat& img, const cv::Point& start, const cv::Point& end, const cv::Scalar& c1,
                   const cv::Scalar& c2) {
  cv::LineIterator iter(img, start, end, cv::LINE_8);

  for (int i = 0; i < iter.count; i++, iter++) {
    double alpha = double(i) / iter.count;
    // note: using img.at<T>(iter.pos()) is faster, but
    // then you have to deal with mat type and channel number yourself
    img(cv::Rect(iter.pos(), cv::Size(1, 1))) = c1 * (1.0 - alpha) + c2 * alpha;
  }
}

void Drawing::polylines(cv::Mat& img, const std::vector<cv::Point>& points, const cv::Scalar& c1,
                        const cv::Scalar& c2) {
  if (!points.size()) {
    return;
  }
  float totalDist = 0.0f;
  for (size_t i = 0; i < points.size() - 1; i++) {
    totalDist += (float)cv::norm(points[i] - points[(i + 1) % points.size()]);
  }
  float a = 0.0f;
  for (size_t i = 0; i < points.size() - 1; i++) {
    cv::Scalar ci_0 = c1 * (1.0 - (a / totalDist)) + c2 * (a / totalDist);
    a += (float)cv::norm(points[i] - points[(i + 1) % points.size()]);
    cv::Scalar ci_1 = c1 * (1.0 - (a / totalDist)) + c2 * (a / totalDist);
    line(img, points[i], points[(i + 1) % points.size()], ci_0, ci_1);
  }
}

}  // namespace Util
}  // namespace VideoStitch
