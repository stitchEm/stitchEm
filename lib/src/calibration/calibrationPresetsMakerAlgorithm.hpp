// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "calibrationPresetsMakerConfig.hpp"

#include "libvideostitch/algorithm.hpp"

namespace VideoStitch {
namespace CalibrationPresetsMaker {

/**
 * @brief Auto-crop for circular fisheye camera
 */
class VS_EXPORT CalibrationPresetsMakerAlgorithm : public Util::Algorithm {
 public:
  static const char* docString;
  explicit CalibrationPresetsMakerAlgorithm(const Ptv::Value* config);
  virtual ~CalibrationPresetsMakerAlgorithm();

 public:
  /**
   @brief Specialization of Algorithm::apply
   @param pano the input/output panorama definition
   @param progress a callback object to give information about the progress of calibration algorithm
   @param ctx An optional context object instance
   @return a description of the result
   */
  virtual Potential<Ptv::Value> apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                      Util::OpaquePtr** ctx = nullptr) const override;

 private:
  CalibrationPresetsMakerConfig calibrationPresetsMakerConfig;
};

}  // namespace CalibrationPresetsMaker
}  // namespace VideoStitch
