// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "keypointExtractor.hpp"

using namespace cv;

namespace VideoStitch {
namespace Calibration {

KeypointExtractor::KeypointExtractor(unsigned int nb_octaves, unsigned int nb_sublevels, double threshold) {
  this->nOctaves = nb_octaves;
  this->nSubLevels = nb_sublevels;
  this->threshold = threshold;
  extractor = cv::AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, (float)threshold, nOctaves, nSubLevels, KAZE::DIFF_PM_G2);
}

Status KeypointExtractor::extract(const cv::Mat& image, KPList& keypoints, DescriptorList& descriptors,
                                  const cv::Mat& mask) const {
  if (!extractor) {
    return {Origin::CalibrationAlgorithm, ErrType::SetupFailure, "Failed to instantiate keypoint detector"};
  }

  /**
  Effectively call extractor
  */
  extractor->detectAndCompute(image, mask, keypoints, descriptors);

  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
