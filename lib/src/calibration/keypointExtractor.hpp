// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef KEYPOINT_EXTRACTOR_HPP_
#define KEYPOINT_EXTRACTOR_HPP_

#include "cvImage.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/logging.hpp"

#include <opencv2/features2d/features2d.hpp>

#include <vector>

namespace VideoStitch {
namespace Calibration {

typedef std::vector<cv::KeyPoint> KPList;
typedef cv::Mat DescriptorList;

/**
 * @brief This class is a wrapper around OpenCV's functions to detect keypoints and extract keypoint descriptors
 */
class KeypointExtractor {
 public:
  /**
  @brief Create extractor
  @param nb_octaves number of octaves to detect into
  @param nb_sublevels number of levels to detect into
  @param threshold threshold value
  */
  explicit KeypointExtractor(unsigned int nb_octaves, unsigned int nb_sublevels, double threshold);

  /**
   * @brief Detect keypoints on given image and compute their descriptors.
   * @return true on success, in which case @keypoints and @descriptors contain respectively the detected keypoints and
   * extracted descriptors
   */
  Status extract(const cv::Mat& image, KPList& keypoints, DescriptorList& descriptors, const cv::Mat& mask) const;

 private:
  cv::Ptr<cv::Feature2D> extractor;
  unsigned int nOctaves;
  unsigned int nSubLevels;
  double threshold;
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
