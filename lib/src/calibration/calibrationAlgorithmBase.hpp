// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "cvImage.hpp"
#include "calibrationConfig.hpp"
#include "gpu/buffer.hpp"

namespace VideoStitch {
namespace Calibration {

class VS_EXPORT CalibrationAlgorithmBase {
 public:
  explicit CalibrationAlgorithmBase(const Ptv::Value* config);
  virtual ~CalibrationAlgorithmBase();

 protected:
  /**
  @brief Convert an RGBA device image into a CvImage
  @note Essentially what it does is convert the image to grayscale and encapsulate the data into an OpenCV-readable
  form.
  */
  Status loadInputImage(std::shared_ptr<CvImage>& result, const GPU::Surface& input, int width, int height) const;

 protected:
  /**
  Keep a copy of the configuration locally
  */
  CalibrationConfig calibConfig;

  /**
  @brief Compute the number of units that will be sent to calibProgress to reach completion
  @param numVideoInputs number of video inputs (i.e. cameras)
  @param numFramesTuples number of frames tuples used for calibration
  @return total number of units that will be reported by the calibration algorithm
  */
  double getProgressUnits(const int numVideoInputs, const int numFramesTuples) const;
};

}  // namespace Calibration
}  // namespace VideoStitch
