// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <util/pngutil.hpp>
#include <util/geometryProcessingUtils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <cassert>
#include <iostream>
#include <memory>
#include <math.h>
#include <random>
#include <string>

//#define DUMP_TEST_RESULT

#if defined(DUMP_TEST_RESULT)
#undef NDEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "../util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Testing {

void testNearestPair() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> di(-100.0f, 100.0f);
  gen.seed(0);

  for (int sampleCount = 1; sampleCount < 500; sampleCount++) {
    cv::Vec2i nearestPair;
    std::vector<cv::Point2f> point0s, point1s;
    const int n = sampleCount;
    const int m = sampleCount;
    for (int i = 0; i < n; i++) {
      point0s.push_back(cv::Point2f(di(gen), di(gen)));
    }
    for (int i = 0; i < m; i++) {
      point1s.push_back(cv::Point2f(di(gen), di(gen)));
    }
    Util::GeometryProcessing::findNearestPair(point0s, point1s, nearestPair);
    double minDist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < point0s.size(); i++) {
      for (size_t j = 0; j < point1s.size(); j++) {
        double dist = cv::norm(point0s[i] - point1s[j]);
        if (dist < minDist) {
          minDist = dist;
        }
      }
    }
    double nearestPairDist = cv::norm(point0s[nearestPair[0]] - point1s[nearestPair[1]]);
    ENSURE(std::abs(nearestPairDist - minDist) < 0.001f);
  }
  std::cout << "*** Test nearest pair passed." << std::endl;
}

Status loadBoundaryCoords(const std::string& filename, int64_t& width, int64_t& height,
                          std::vector<cv::Point2f>& points) {
  std::vector<unsigned char> data;
  Util::PngReader::readRGBAFromFile(filename.c_str(), width, height, data);
  cv::Mat image(cv::Size((int)width, (int)height), CV_8UC4, &data[0]);

  const int thresh = 100;
  cv::Mat src_gray;
  cv::Mat canny_output;
  cv::cvtColor(image, src_gray, cv::COLOR_RGBA2GRAY);
  // Detect edges using canny
  cv::Canny(src_gray, canny_output, thresh, thresh * 2, 3);
  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_output, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  points.clear();
  if (contours.size() > 0) {
    for (size_t i = 0; i < contours[0].size(); i++) {
      points.push_back(contours[0][i]);
    }
    contours.resize(1);
  }
  return Status::OK();
}

void testContourMatching() {
#ifdef DUMP_TEST_RESULT
  std::string workingPath =
      "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/";
#else
  std::string workingPath = "";
#endif

  std::vector<float> matchingCosts = {0.127777740f, 0.835167468f};
  for (int test = 0; test <= 1; test++) {
    std::vector<cv::Point2f> point0s, point1s;
    std::vector<int> matchIndices;
    int64_t width0, height0, width1, height1;
    loadBoundaryCoords(workingPath + "data/contourmatching/" + std::to_string(test) + "-a.png", width0, height0,
                       point0s);
    loadBoundaryCoords(workingPath + "data/contourmatching/" + std::to_string(test) + "-b.png", width1, height1,
                       point1s);
    float cost = Util::GeometryProcessing::contourMatching(point0s, point1s, matchIndices);
#ifdef DUMP_TEST_RESULT
    std::cout << "*** Test contour " << cost << " passed." << std::endl;
#else
    ENSURE(std::abs(cost - matchingCosts[test]) < 0.001f, "Matching costs do not match");
#endif
    std::cout << "*** Test contour " << test << " passed." << std::endl;
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::testNearestPair();
  VideoStitch::Testing::testContourMatching();

  return 0;
}
