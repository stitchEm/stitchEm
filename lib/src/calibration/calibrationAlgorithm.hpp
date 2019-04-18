// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CALIBRATION_ALGORITHM_HPP_
#define CALIBRATION_ALGORITHM_HPP_

#include "calibrationAlgorithmBase.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/status.hpp"

namespace VideoStitch {
namespace Calibration {

class CalibrationProgress;

/**
@brief Instance of Algorithm.
@details This is a communication layer between the calibration object and the apps
*/
class VS_EXPORT CalibrationAlgorithm : public Util::Algorithm, public CalibrationAlgorithmBase {
 public:
  static const char* docString;
  explicit CalibrationAlgorithm(const Ptv::Value* config);
  virtual ~CalibrationAlgorithm();

 public:
  /**
  @brief Specialization of Algorithm::apply
  @param pano the input/output panorama definition
  @param progress a callback object to give information about the progress of calibration algorithm
  @param ctx An optional context object instance
  @return a description of the result
  */
  Potential<Ptv::Value> apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                              Util::OpaquePtr** ctx = nullptr) const override;

 protected:
  /**
  @brief Retrieves images for calibration
  @param rig the output images
  @param pano the input panorama definition
  @param progress the progress reporter
  @return a status
  */
  Status retrieveImages(RigCvImages& rig, const Core::PanoDefinition& pano, CalibrationProgress& progress) const;
};

}  // namespace Calibration
}  // namespace VideoStitch
#endif
