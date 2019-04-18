// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/config.hpp"

namespace VideoStitch {
namespace Util {

class PhotometricCalibrationProblem;

/**
 * An algorithm that estimates vignetting parameters and EMoR in one step.
 */
class PhotometricCalibrationBase {
 public:
  explicit PhotometricCalibrationBase(const Ptv::Value* config);
  virtual ~PhotometricCalibrationBase() {}

 protected:
  int maxSampledPoints;
  int minPointsPerInput;
  int neighbourhoodSize;
};

class PhotometricCalibrationAlgorithm : public Algorithm, public PhotometricCalibrationBase {
 public:
  /**
   * The algo docstring.
   */
  static const char* docString;
  explicit PhotometricCalibrationAlgorithm(const Ptv::Value* config);
  virtual ~PhotometricCalibrationAlgorithm() {}

  Potential<Ptv::Value> apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                              OpaquePtr** = NULL) const override;

 private:
  int firstFrame = 0;
  int lastFrame = 0;
};

}  // namespace Util
}  // namespace VideoStitch
