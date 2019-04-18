// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "libvideostitch/config.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"

#include "gpu/stream.hpp"
#include "gpu/uniqueBuffer.hpp"

#include <opencv2/imgproc.hpp>

namespace VideoStitch {
namespace Util {

class VS_EXPORT GeometryProcessing {
 public:
  static int norm2(const cv::Vec3b a, const cv::Vec3b b);
  static int norm2(const unsigned char a, const unsigned char b);
  static int norm2(const cv::Point a, const cv::Point b);
  static float norm2(const cv::Point2f a, const cv::Point2f b);
  static float length(const cv::Point a, const cv::Point b);
  static float length(const cv::Point2f a, const cv::Point2f b);
  static float lengthSqr(const cv::Point2f a, const cv::Point2f b);

  /**
   * @brief Check whether a point is on the border of the input image
   */
  static bool onBorder(const cv::Point point, const cv::Size size);

  /**
   * @brief Check whether a point stays inside an image
   */
  static bool insideImage(const cv::Point& point, const cv::Size& imageSize);

  /**
   * @brief Check whether a point stays inside an image
   */
  static bool insideImage(const cv::Point& point, const cv::Mat& image);

  /**
   * @brief Check whether a point stays inside a circle
   */
  static bool pointInsideCircle(const cv::Point2d p, const cv::Point3d circle);

  /**
   * @brief Check whether a vector contains a point
   */
  static bool pointInVector(const cv::Point p, const std::vector<cv::Point>* points);

  /**
   * @brief Sample points uniformly on a circle inside the image
   */
  static void getUniformSampleOnCircle(const size_t sampleCount, const cv::Size imageSize,
                                       const cv::Point3d& inputCircle, std::vector<cv::Point>& points);

  /**
   * @brief Sample points uniformly on a polygon
   */
  static std::vector<cv::Point2f> getUniformSampleOnPolygon(const std::vector<cv::Point>& polygon,
                                                            const size_t sampleCount);

  /**
   * @brief Find contour of a certain bit
   */
  template <typename T>
  static Status findImageContours(const cv::Size size, const std::vector<T>& data, const T bitMask,
                                  std::vector<std::vector<cv::Point>>& points, const int method = CV_CHAIN_APPROX_NONE);

  /**
   * @brief Draw polygon to an output buffer
   */
  static Status drawPolygon(const cv::Size size, const std::vector<std::vector<cv::Point>>& points,
                            std::vector<unsigned char>& mask);
  static Status drawPolygon(const cv::Size size, const std::vector<cv::Point2f>& points,
                            std::vector<unsigned char>& mask);
  static Status drawPolygon(const cv::Size size, const std::vector<cv::Point>& points,
                            std::vector<unsigned char>& mask);

  static std::vector<std::vector<cv::Point>> convertPoint2fToPoint(const std::vector<cv::Point2f>& points);

  /**
   * @brief Draw the correspondent points of two contours, and dump the result to a file
   */
  static Status dumpContourMatch(const std::string& filename, const int width, const int height,
                                 const std::vector<cv::Point2f>& point0s, const std::vector<cv::Point2f>& point1s,
                                 const std::vector<int>& matchIndices);

  /**
   * @brief Draw a contour and dump the result to a file
   */
  static Status dumpContourToPngFile(const std::string& filename, const int width, const int height,
                                     const std::vector<std::vector<cv::Point>>& contours);

  /**
   * @brief Draw a set of point and dump the result to a file
   */
  static Status dumpPoints(const std::string& filename, const int width, const int height,
                           const std::vector<cv::Point2f>& points);

  /**
   * @brief Calculate the perimeter of a simple polygon
   */
  static float getPolygonPerimeter(const std::vector<cv::Point>& polygon);

  /**
   * @brief Find the bounding rect of a set of points
   */
  static cv::Rect2f getBoundingRect(const std::vector<cv::Point2f>& points);

  /**
   * @brief Perform contour matching
   */
  static float contourMatching(const std::vector<cv::Point2f>& point0s, const std::vector<cv::Point2f>& point1s,
                               std::vector<int>& matchIndices, const bool nearestStartingPoint = true);

  /**
   * @brief Find the nearest pair from two contours
   */
  static Status findNearestPair(const std::vector<cv::Point2f>& point0s, const std::vector<cv::Point2f>& point1s,
                                cv::Vec2i& nearestPair);

 private:
  static float findNearestPair(const int l, const int r, const std::vector<cv::Vec2i>& setCounts,
                               const std::vector<std::tuple<int, int, cv::Point2f>>& points, cv::Vec2i& nearestPair);

  static bool isValidRange(const int l, const int r, const std::vector<cv::Vec2i>& setCounts);

  static float contourMatching(const int x, const std::vector<cv::Point2f>& point0s, const int y,
                               const std::vector<cv::Point2f>& point1s, std::vector<int>& matchIndices);
};

}  // namespace Util
}  // namespace VideoStitch
