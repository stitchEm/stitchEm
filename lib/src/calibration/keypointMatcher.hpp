// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "keypointExtractor.hpp"

#include "libvideostitch/controlPointListDef.hpp"

namespace VideoStitch {
namespace Calibration {

/**
 * @brief This class is a wrapper around OpenCV's keypoint matcher
 */
class KeypointMatcher {
 public:
  /**
  @brief Constructor
  @param ratio minimum distance ratio allowed
  @param crossCheck applies cross-check between the matches, they will be each other's best match
  */
  explicit KeypointMatcher(const double ratio, const bool crossCheck = true, const bool bruteForce = true);

  /**
   * @brief Finds control points between images A and B by matching the given descriptors.
   * @return true on success, in which case @matchedControlPoints contains the list of computed keypoints
   */
  Status match(const unsigned int frameNumber, const std::pair<unsigned int, unsigned int>& inputPair,
               const KPList& keypointsA, const DescriptorList& descriptorsA, const KPList& keypointsB,
               const DescriptorList& descriptorsB, Core::ControlPointList& matchedControlPoints) const;

 private:
  cv::Ptr<cv::DescriptorMatcher> matcher;
  double nndrRatio;
  bool crossCheck;
};

}  // namespace Calibration
}  // namespace VideoStitch
