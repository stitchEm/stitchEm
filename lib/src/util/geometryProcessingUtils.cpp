// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "geometryProcessingUtils.hpp"
#include "pngutil.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <random>
#include <functional>

#ifndef NDEBUG
#include "debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Util {

int GeometryProcessing::norm2(const cv::Vec3b a, const cv::Vec3b b) {
  const int diffX = int(a[0]) - int(b[0]);
  const int diffY = int(a[1]) - int(b[1]);
  const int diffZ = int(a[2]) - int(b[2]);
  return diffX * diffX + diffY * diffY + diffZ * diffZ;
}

int GeometryProcessing::norm2(const cv::Point a, const cv::Point b) {
  const int diffX = a.x - b.x;
  const int diffY = a.y - b.y;
  return diffX * diffX + diffY * diffY;
}

float GeometryProcessing::norm2(const cv::Point2f a, const cv::Point2f b) {
  const float diffX = a.x - b.x;
  const float diffY = a.y - b.y;
  return diffX * diffX + diffY * diffY;
}

int GeometryProcessing::norm2(const unsigned char a, const unsigned char b) {
  int diff = int(a) - int(b);
  return diff * diff;
}

float GeometryProcessing::length(const cv::Point a, const cv::Point b) { return std::sqrt((float)norm2(a, b)); }

float GeometryProcessing::lengthSqr(const cv::Point2f a, const cv::Point2f b) { return (float)norm2(a, b); }

float GeometryProcessing::length(const cv::Point2f a, const cv::Point2f b) { return (float)std::sqrt(lengthSqr(a, b)); }

bool GeometryProcessing::insideImage(const cv::Point& point, const cv::Mat& image) {
  return (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows);
}

bool GeometryProcessing::insideImage(const cv::Point& point, const cv::Size& imageSize) {
  return (point.x >= 0 && point.x < imageSize.width && point.y >= 0 && point.y < imageSize.height);
}

bool GeometryProcessing::pointInsideCircle(const cv::Point2d p, const cv::Point3d circle) {
  const double dx_sqr = (p.x - circle.x) * (p.x - circle.x);
  const double dy_sqr = (p.y - circle.y) * (p.y - circle.y);
  return dx_sqr + dy_sqr <= circle.z * circle.z;
}

bool GeometryProcessing::onBorder(const cv::Point point, const cv::Size size) {
  if (point.x != 0 && point.x != size.width - 1 && point.y != 0 && point.y != size.height - 1) {
    return false;
  } else {
    return true;
  }
}

bool GeometryProcessing::pointInVector(const cv::Point p, const std::vector<cv::Point>* points) {
  if (points == nullptr) {
    return false;
  }
  for (auto point : *points) {
    if (p == point) {
      return true;
    }
  }
  return false;
}

void GeometryProcessing::getUniformSampleOnCircle(const size_t sampleCount, const cv::Size imageSize,
                                                  const cv::Point3d& inputCircle, std::vector<cv::Point>& points) {
  // http://mathworld.wolfram.com/CirclePointPicking.html
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0);
  std::uniform_real_distribution<double> disX1(-1.0, 1.0);
  std::uniform_real_distribution<double> disX2(-1.0, 1.0);
  while (points.size() < sampleCount) {
    const double x1 = disX1(gen);
    const double x2 = disX2(gen);
    const double sum_sq = x1 * x1 + x2 * x2;
    if (sum_sq < 1.0) {
      const double x = (x1 * x1 - x2 * x2) / sum_sq;
      const double y = 2 * x1 * x2 / sum_sq;
      const cv::Point scaledPoint((int)(x * inputCircle.z + inputCircle.x), (int)(y * inputCircle.z + inputCircle.y));
      if (Util::GeometryProcessing::insideImage(scaledPoint, imageSize)) {
        points.push_back(scaledPoint);
      }
    }
  }
}

std::vector<cv::Point2f> GeometryProcessing::getUniformSampleOnPolygon(const std::vector<cv::Point>& polygon,
                                                                       const size_t sampleCount) {
  std::vector<cv::Point2f> sampledPoints;
  if (polygon.size() < 3) {
    return sampledPoints;
  }
  float polygonSize = getPolygonPerimeter(polygon);
  float s = 0.0f;
  float dx = polygonSize / (sampleCount - 1);
  size_t index = 0;
  size_t i = 0;
  float x = 0.0f;
  while (i < sampleCount && index < polygon.size()) {
    const float t = s;
    s += length(polygon[index], polygon[(index + 1) % polygon.size()]);
    while (x <= s && i < sampleCount) {
      cv::Point2f sampledPoint = cv::Point2f(polygon[index]) * (1.0f - (x - t) / (s - t)) +
                                 cv::Point2f(polygon[(index + 1) % polygon.size()]) * ((x - t) / (s - t));
      sampledPoints.push_back(sampledPoint);
      x += dx;
      i++;
    }
    index++;
  }
  return sampledPoints;
}

Status GeometryProcessing::findNearestPair(const std::vector<cv::Point2f>& point0s,
                                           const std::vector<cv::Point2f>& point1s, cv::Vec2i& nearestPair) {
  // https://en.wikipedia.org/wiki/Closest_pair_of_points_problem

  if (!point0s.size() || !point1s.size()) {
    return {Origin::GeometryProcessingUtils, ErrType::InvalidConfiguration, "One of the input arrays is empty."};
  }

  nearestPair = cv::Vec2i(-1, -1);
  std::vector<std::tuple<int, int, cv::Point2f>> points;
  std::vector<cv::Vec2i> setCounts;
  for (size_t i = 0; i < point0s.size(); i++) {
    points.push_back(std::make_tuple(0, (int)i, point0s[i]));
  }
  for (size_t i = 0; i < point1s.size(); i++) {
    points.push_back(std::make_tuple(1, (int)i, point1s[i]));
  }

  // Sort increasing of x
  std::sort(points.begin(), points.end(),
            [](const std::tuple<int, int, cv::Point2f>& a, const std::tuple<int, int, cv::Point2f>& b) -> bool {
              return std::get<2>(a).x < std::get<2>(b).x;
            });

  if (points.size() > 0) {
    setCounts.push_back(cv::Vec2i(0, 0));
    for (size_t i = 0; i < points.size(); i++) {
      const cv::Vec2i lastCount = setCounts.back();
      if (std::get<0>(points[i]) == 0) {
        setCounts.push_back(cv::Vec2i(lastCount[0] + 1, lastCount[1]));
      } else {
        setCounts.push_back(cv::Vec2i(lastCount[0], lastCount[1] + 1));
      }
    }
  }
  cv::Vec2i indices;
  findNearestPair(0, (int)points.size() - 1, setCounts, points, indices);
  nearestPair = cv::Vec2i(std::get<1>(points[indices[0]]), std::get<1>(points[indices[1]]));
  return Status::OK();
}

bool GeometryProcessing::isValidRange(const int l, const int r, const std::vector<cv::Vec2i>& setCounts) {
  if (l >= r || l < 0 || r + 1 >= ((int)setCounts.size())) {
    return false;
  }
  if (setCounts[r + 1][0] == setCounts[l][0] || setCounts[r + 1][1] == setCounts[l][1]) {
    return false;
  }
  return true;
}

float GeometryProcessing::findNearestPair(const int l, const int r, const std::vector<cv::Vec2i>& setCounts,
                                          const std::vector<std::tuple<int, int, cv::Point2f>>& points,
                                          cv::Vec2i& nearestPair) {
  // There is only 1 single set in this range, can't compute the min distance
  if (!isValidRange(l, r, setCounts)) {
    return -1;
  }
  if (r == l + 1) {
    if (std::get<0>(points[l]) != std::get<0>(points[r])) {
      nearestPair[std::get<0>(points[l])] = l;
      nearestPair[std::get<0>(points[r])] = r;
      return (float)cv::norm(std::get<2>(points[l]) - std::get<2>(points[r]));
    } else {
      return -1;
    }
  }
  // Make sure that divide the set into 2 and at least 1 of them is valid
  int m = -1;
  for (int k = (l + r) / 2 - 1; k <= (l + r) / 2 + 1; k++) {
    if (isValidRange(l, k, setCounts) || isValidRange(k + 1, r, setCounts)) {
      m = k;
      break;
    }
  }
  if (m < 0) {
    return -1;
  }

  cv::Vec2i nearestPairLeft;
  const float minLeft = findNearestPair(l, m, setCounts, points, nearestPairLeft);
  cv::Vec2i nearestPairRight;
  const float minRight = findNearestPair(m + 1, r, setCounts, points, nearestPairRight);

  float minDistance = std::numeric_limits<float>::max();

  if (minRight < 0 || (minLeft < minRight && minLeft >= 0)) {
    minDistance = minLeft;
    nearestPair = nearestPairLeft;
  } else {
    minDistance = minRight;
    nearestPair = nearestPairRight;
  }

  // Find the min distance from a point on the left side to a point on the right side
  int i = m;
  while ((i >= l) && (std::get<2>(points[m]).x - std::get<2>(points[i]).x < minDistance)) {
    int j = m + 1;
    while ((j <= r) && (std::get<2>(points[j]).x - std::get<2>(points[m]).x < minDistance)) {
      if (std::get<0>(points[i]) != std::get<0>(points[j])) {
        float dist = (float)cv::norm(std::get<2>(points[i]) - std::get<2>(points[j]));
        if (dist < minDistance) {
          minDistance = dist;
          nearestPair[std::get<0>(points[i])] = i;
          nearestPair[std::get<0>(points[j])] = j;
        }
      }
      j++;
    }
    i--;
  }
  return minDistance;
}

cv::Rect2f GeometryProcessing::getBoundingRect(const std::vector<cv::Point2f>& points) {
  if (points.size() == 0) {
    return cv::Rect2f(0, 0, 0, 0);
  }
  cv::Point2f pointMin = points[0];
  cv::Point2f pointMax = points[0];
  for (auto point : points) {
    if (point.x < pointMin.x) pointMin.x = point.x;
    if (point.y < pointMin.y) pointMin.y = point.y;
    if (point.x > pointMax.x) pointMax.x = point.x;
    if (point.y > pointMax.y) pointMax.y = point.y;
  }
  return cv::Rect2f(pointMin.x, pointMin.y, pointMax.x - pointMin.x, pointMax.y - pointMin.y);
}

float GeometryProcessing::contourMatching(const int x, const std::vector<cv::Point2f>& point0sOld, const int y,
                                          const std::vector<cv::Point2f>& point1sOld, std::vector<int>& matchIndices) {
  // Shift both array to x and y accordingly
  std::vector<cv::Point2f> point0s;
  point0s.insert(point0s.end(), point0sOld.begin() + x, point0sOld.end());
  if (x > 0) {
    point0s.insert(point0s.end(), point0sOld.begin(), point0sOld.begin() + x);
  }

  std::vector<cv::Point2f> point1s;
  point1s.insert(point1s.end(), point1sOld.begin() + y, point1sOld.end());
  if (y > 0) {
    point1s.insert(point1s.end(), point1sOld.begin(), point1sOld.begin() + y);
  }

  // Get the bounding box of the points
  cv::Rect2f rect = getBoundingRect(point0s) | getBoundingRect(point1s);

  std::function<cv::Point2f(cv::Point2f)> scaleFunction = [&rect](cv::Point2f p) {
    return cv::Point2f((p.x - rect.x) / rect.width, (p.y - rect.y) / rect.height);
  };

  // Normalize the points' coordinate
  std::transform(point0s.begin(), point0s.end(), point0s.begin(), scaleFunction);
  std::transform(point1s.begin(), point1s.end(), point1s.begin(), scaleFunction);

  // Find the correspondence points
  std::vector<float> costs(point1s.size());
  std::vector<float> prevCosts(point1s.size());
  std::vector<float> contour1_distance(point1s.size());
  std::vector<std::vector<int>> prevMatch;
  for (size_t i = 0; i < point0s.size(); i++) {
    prevMatch.push_back(std::vector<int>(point1s.size(), -1));
  }

  contour1_distance[0] = 0.0f;
  for (size_t i = 1; i < point1s.size(); i++) {
    contour1_distance[i] = contour1_distance[i - 1] + length(point1s[i], point1s[i - 1]);
  }

  std::fill(prevCosts.begin(), prevCosts.end(), std::numeric_limits<float>::max());
  prevCosts[0] = (float)cv::norm(point0s[0] - point1s[0]);

  // Use dynamic programming to find the best boundary matching from contour 0 in contour 1
  // For better performance, this part should be implemented in CUDA
  const float same_pair_cost = 5.0f;  // This cost is used to penalize assigning the same point in contour 1
  const float stride_cost =
      0.01f;  // This cost is used to penalize assigning "distance" points in contour 1 to neighbor points in contour 0.

  for (size_t i = 1; i < point0s.size(); i++) {
    /*
    NOTE: This is the full version of the optimized code below.
          The two should give identical results
    const int maxDistance = 20;
    for (int j = 0; j < (int) point1s.size(); j++) {
      costs[j] = prevCosts[j] + same_pair_cost;
      prevMatch[i][j] = (int) j;
      const int minIndex = std::max( 0, j - maxDistance);
      for (int t = j - 1; t >= minIndex; t--) {
        const float cost = prevCosts[t] + stride_cost*(contour1_distance[j] - contour1_distance[t]);
        if (cost < costs[j]) {
          costs[j] = cost;
          prevMatch[i][j] = t;
        }
      }
      costs[j] += length(point0s[i], point1s[j]);
    }
    */

    /* This part is the optimized version of the commented code above */
    size_t bestPrevJ = 0;
    for (size_t j = 0; j < point1s.size(); j++) {
      // Update best cost
      float bestPrevCost = prevCosts[bestPrevJ] + stride_cost * (contour1_distance[j] - contour1_distance[bestPrevJ]);
      if (j >= 1) {
        float currentCost = prevCosts[j - 1] + stride_cost * (contour1_distance[j] - contour1_distance[j - 1]);
        if (currentCost < bestPrevCost) {
          bestPrevCost = currentCost;
          bestPrevJ = j - 1;
        }
      }
      float sameCost = prevCosts[j] + same_pair_cost;
      if (sameCost < bestPrevCost || bestPrevJ == j) {
        costs[j] = sameCost + length(point0s[i], point1s[j]);
        prevMatch[i][j] = (int)j;
      } else {
        costs[j] = bestPrevCost + length(point0s[i], point1s[j]);
        prevMatch[i][j] = (int)bestPrevJ;
      }
    }
    prevCosts = costs;
  }

  // Find the path by tracing back
  if (point0s.size()) {
    matchIndices.resize(point0s.size());
    int j = (int)point1s.size() - 1;
    for (int i = (int)point0s.size() - 1; i >= 0; i--) {
      matchIndices[(x + i) % point0s.size()] = (y + j) % point1s.size();
      if (prevMatch[i][j]) {
        j = prevMatch[i][j];
      }
    }
  }
  return costs[point1s.size() - 1] / point0s.size();
}

float GeometryProcessing::contourMatching(const std::vector<cv::Point2f>& point0s,
                                          const std::vector<cv::Point2f>& point1s, std::vector<int>& matchIndices,
                                          const bool nearestStartingPoint) {
  // Find the two nearest points, they are the starting point for both contour
  if (nearestStartingPoint) {
    cv::Vec2i nearestPair;
    findNearestPair(point0s, point1s, nearestPair);
    const int x = nearestPair[0];
    const int y = nearestPair[1];
    return contourMatching(x, point0s, y, point1s, matchIndices);
  } else {
    return std::numeric_limits<float>::max();
  }
}

template <typename T>
Status GeometryProcessing::findImageContours(const cv::Size size, const std::vector<T>& data, const T bitMask,
                                             std::vector<std::vector<cv::Point>>& contours, const int method) {
  if (size.width * size.height != (int)data.size()) {
    return {Origin::Input, ErrType::InvalidConfiguration, "Corrupted input for finding image contour"};
  }
  std::vector<unsigned char> image;
  for (size_t i = 0; i < data.size(); i++) {
    image.push_back((data[i] & bitMask) ? 255 : 0);
  }
  cv::Mat src_gray(size, CV_8UC1, &image[0]);
  /// Find contours
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(src_gray, contours, hierarchy, cv::RETR_EXTERNAL, method, cv::Point(0, 0));
  return Status::OK();
}

template Status GeometryProcessing::findImageContours<uint32_t>(const cv::Size size, const std::vector<uint32_t>& data,
                                                                const uint32_t bitMask,
                                                                std::vector<std::vector<cv::Point>>& contours,
                                                                const int method);

template Status GeometryProcessing::findImageContours<unsigned char>(const cv::Size size,
                                                                     const std::vector<unsigned char>& data,
                                                                     const unsigned char bitMask,
                                                                     std::vector<std::vector<cv::Point>>& contours,
                                                                     const int method);

Status GeometryProcessing::drawPolygon(const cv::Size size, const std::vector<cv::Point2f>& points,
                                       std::vector<unsigned char>& mask) {
  return drawPolygon(size, convertPoint2fToPoint(points), mask);
}

Status GeometryProcessing::drawPolygon(const cv::Size size, const std::vector<cv::Point>& points,
                                       std::vector<unsigned char>& mask) {
  std::vector<std::vector<cv::Point>> round_points;
  round_points.push_back(points);
  return drawPolygon(size, round_points, mask);
}

Status GeometryProcessing::drawPolygon(const cv::Size size, const std::vector<std::vector<cv::Point>>& points,
                                       std::vector<unsigned char>& mask) {
  cv::Mat img(size, CV_8UC1);
  img.setTo(0);
  std::vector<cv::Point*> contours(points.size());
  std::vector<int> countours_n(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    contours[i] = (cv::Point*)(&points[i][0]);
    countours_n[i] = (int)points[i].size();
  }

  cv::fillPoly(img, (const cv::Point**)&contours[0], (const int*)&countours_n[0], (int)points.size(), cv::Scalar(255));
  // @NOTE: This is a hack. I still don't why openCV draws nothing around the image boundaries
  //  As a hack, I draw all the poly-line with width = 2.
  //  --> All polygon's boundaries are 1 pixel bigger than they actually are
  cv::polylines(img, (const cv::Point**)&contours[0], (const int*)&countours_n[0], (int)points.size(), true,
                cv::Scalar(255), 2);
  mask.resize(size.width * size.height, 0);
  for (int j = 0; j < img.cols; j++)
    for (int i = 0; i < img.rows; i++) {
      mask[i * size.width + j] = img.at<unsigned char>(cv::Point(j, i));
    }
  return Status::OK();
}

Status GeometryProcessing::dumpContourMatch(const std::string& filename, const int width, const int height,
                                            const std::vector<cv::Point2f>& point0s,
                                            const std::vector<cv::Point2f>& point1s,
                                            const std::vector<int>& matchIndices) {
  cv::Mat drawing = cv::Mat::zeros(height, width, CV_8UC4);
  for (size_t i = 0; i < matchIndices.size(); i++) {
    if (i % 3 == 0) {
      cv::line(drawing, point0s[i], point1s[matchIndices[i]], cv::Scalar(255, 255, 255, 255), 1);
    }
    cv::circle(drawing, point0s[i], 2, cv::Scalar(255, 0, 0, 255));
    cv::circle(drawing, point1s[matchIndices[i]], 2, cv::Scalar(0, 255, 0, 255));
  }
  if (!Util::PngReader::writeRGBAToFile(filename.c_str(), width, height, drawing.data)) {
    return {Origin::GeometryProcessingUtils, ErrType::RuntimeError, "Failed to dump result to file"};
  }
  return Status::OK();
}

std::vector<std::vector<cv::Point>> GeometryProcessing::convertPoint2fToPoint(const std::vector<cv::Point2f>& points) {
  std::vector<std::vector<cv::Point>> round_points;
  std::vector<cv::Point> round_point;
  for (size_t i = 0; i < points.size(); i++) {
    round_point.push_back(cv::Point(points[i]));
  }
  round_points.push_back(round_point);
  return round_points;
}

float GeometryProcessing::getPolygonPerimeter(const std::vector<cv::Point>& polygon) {
  float p = 0.0f;
  for (size_t i = 0; i < polygon.size(); i++) {
    p += length(polygon[i], polygon[(i + 1) % polygon.size()]);
  }
  return p;
}

Status GeometryProcessing::dumpPoints(const std::string& filename, const int width, const int height,
                                      const std::vector<cv::Point2f>& points) {
  cv::Mat drawing = cv::Mat::zeros(height, width, CV_8UC1);
  for (size_t i = 0; i < points.size(); i++) {
    cv::Scalar color = cv::Scalar((unsigned)(255.0f * i / points.size()));
    cv::circle(drawing, cv::Point(points[i]), 1, color);
  }

  cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
  applyColorMap(drawing, image, cv::COLORMAP_JET);

  if (!Util::PngReader::writeBGRToFile(filename.c_str(), width, height, image.data)) {
    return {Origin::GeometryProcessingUtils, ErrType::RuntimeError, "Failed to dump result to file"};
  }
  return Status::OK();
}

Status GeometryProcessing::dumpContourToPngFile(const std::string& filename, const int width, const int height,
                                                const std::vector<std::vector<cv::Point>>& contours) {
  /// Draw contours
  cv::Mat drawing = cv::Mat::zeros(height, width, CV_8UC1);
  for (size_t i = 0; i < contours.size(); i++) {
    for (size_t j = 0; j < contours[i].size(); j++) {
      cv::line(drawing, contours[i][j], contours[i][(j + 1) % contours[i].size()], 3);
    }
  }

  cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
  applyColorMap(drawing, image, cv::COLORMAP_JET);

  if (!Util::PngReader::writeBGRToFile(filename.c_str(), width, height, image.data)) {
    return {Origin::GeometryProcessingUtils, ErrType::RuntimeError, "Failed to dump result to file"};
  }
  return Status::OK();
}

}  // namespace Util
}  // namespace VideoStitch
