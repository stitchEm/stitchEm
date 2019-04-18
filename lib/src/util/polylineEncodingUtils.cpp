// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "polylineEncodingUtils.hpp"
#include "geometryProcessingUtils.hpp"

namespace VideoStitch {
namespace Util {

// https://developers.google.com/maps/documentation/utilities/polylinealgorithm
void PolylineEncoding::polylineEncode(const std::vector<cv::Point2f>& polyPoints, std::string& encoded,
                                      const float scale) {
  cv::Point2f lastPoint(0.0f, 0.0f);
  encoded = std::string("");
  for (size_t i = 0; i < polyPoints.size(); i++) {
    cv::Point2f diff = polyPoints[i] - lastPoint;
    const int32_t lat = (int32_t)std::round(diff.x * scale);
    const int32_t lng = (int32_t)std::round(diff.y * scale);
    encoded += polylineEncodeValue(lat);
    encoded += polylineEncodeValue(lng);
    lastPoint = polyPoints[i];
  }
}

void PolylineEncoding::polylineDecode(const std::string& encoded, std::vector<cv::Point2f>& polyPoints,
                                      const float scale) {
  int32_t index = 0;
  int32_t currentLat = 0;
  int32_t currentLng = 0;
  int32_t next5bits = 0;
  const int32_t encodedLength = (int32_t)encoded.length();
  polyPoints.clear();
  while (index < encodedLength) {
    // calculate next latitude
    int32_t decoded = polylineDecodeValue(encoded, index, next5bits);
    if (index >= encodedLength) {
      break;
    }
    currentLat += decoded;

    // calculate next longitude
    decoded = polylineDecodeValue(encoded, index, next5bits);
    if (index >= encodedLength && next5bits >= 32) {
      break;
    }
    currentLng += decoded;

    polyPoints.push_back(cv::Point2f(float(currentLat) / scale, float(currentLng) / scale));
  }
}

int32_t PolylineEncoding::polylineDecodeValue(const std::string& encoded, int32_t& index, int32_t& next5bits) {
  const char* polylineChars = encoded.c_str();
  int32_t sum = 0;
  int32_t shifter = 0;
  const int32_t encodedLength = (int32_t)encoded.length();
  do {
    next5bits = (int)polylineChars[index++] - 63;
    sum |= (next5bits & 31) << shifter;
    shifter += 5;
  } while (next5bits >= 32 && index < encodedLength);

  return (sum & 1) == 1 ? ~(sum >> 1) : (sum >> 1);
}

std::string PolylineEncoding::polylineEncodeValue(const int32_t value) {
  std::string str;
  int32_t shifted = value << 1;
  if (value < 0) {
    shifted = ~shifted;
  }
  int32_t rem = shifted;
  while (rem >= 32) {
    str += (char)((32 | (rem & 31)) + 63);
    rem >>= 5;
  }
  str += (char)(rem + 63);
  return str;
}

void PolylineEncoding::polylineEncode(const std::vector<cv::Point>& polyPoints, std::string& encoded) {
  std::vector<cv::Point2f> points;
  for (size_t i = 0; i < polyPoints.size(); i++) {
    points.push_back(polyPoints[i]);
  }
  polylineEncode(points, encoded, 1.0f);
}

void PolylineEncoding::polylineDecode(const std::string& encoded, std::vector<cv::Point>& polyPoints) {
  std::vector<cv::Point2f> points;
  polylineDecode(encoded, points, 1.0f);
  for (size_t i = 0; i < points.size(); i++) {
    polyPoints.push_back(points[i]);
  }
}

void PolylineEncoding::polylineDecodePolygon(const std::string& values, std::vector<std::vector<cv::Point>>& points) {
  // Decode vector info
  std::vector<int> encodedSizes;
  int32_t index = 0, next5bits = 0;
  int32_t vectorSize = PolylineEncoding::polylineDecodeValue(values, index, next5bits);
  for (int32_t i = 0; i < vectorSize; i++) {
    int32_t val = PolylineEncoding::polylineDecodeValue(values, index, next5bits);
    encodedSizes.push_back(val);
  }
  int32_t offset = index;
  // Decode all components
  for (size_t i = 0; i < encodedSizes.size(); i++) {
    std::vector<cv::Point> reconstructedPoints;
    const std::string component_value = values.substr(offset, encodedSizes[i]);
    Util::PolylineEncoding::polylineDecode(component_value, reconstructedPoints);
    points.push_back(reconstructedPoints);
    offset += encodedSizes[i];
  }
}

void PolylineEncoding::polylineDecodePolygons(const std::map<videoreaderid_t, std::string>& encodedValues,
                                              std::map<videoreaderid_t, std::vector<cv::Point>>& decodedPolygons) {
  decodedPolygons.clear();
  for (auto& encodedValue : encodedValues) {
    const videoreaderid_t imId = encodedValue.first;
    std::vector<std::vector<cv::Point>> points;
    polylineDecodePolygon(encodedValue.second, points);
    decodedPolygons[imId] = points[0];
  }
}

}  // namespace Util
}  // namespace VideoStitch
