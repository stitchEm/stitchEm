// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/parse.hpp"

#include <memory>

namespace VideoStitch {
namespace CalibrationPresetsMaker {

/**
 * @brief Configuration used by the CalibrationPresetsMaker
 */
class VS_EXPORT CalibrationPresetsMakerConfig {
 public:
  explicit CalibrationPresetsMakerConfig(const Ptv::Value* config);

  ~CalibrationPresetsMakerConfig() = default;

  CalibrationPresetsMakerConfig(const CalibrationPresetsMakerConfig&);

  bool isValid() const { return isConfigValid; }

  std::string getPresetsName() const { return presetsName; }

  double getFocalStdDevValuePercentage() const { return focalStdDevValuePercentage; }

  double getCenterStdDevWidthPercentage() const { return centerStdDevWidthPercentage; }

  double getDistortStdDevValuePercentage() const { return distortStdDevValuePercentage; }

  double getYawStdDev() const { return yawStdDev; }

  double getPitchStdDev() const { return pitchStdDev; }

  double getRollStdDev() const { return rollStdDev; }

  double getTranslationXStdDev() const { return translationXStdDev; }

  double getTranslationYStdDev() const { return translationYStdDev; }

  double getTranslationZStdDev() const { return translationZStdDev; }

 private:
  bool isConfigValid;

  std::string presetsName;
  double focalStdDevValuePercentage;
  double centerStdDevWidthPercentage;
  double distortStdDevValuePercentage;
  double yawStdDev;
  double pitchStdDev;
  double rollStdDev;
  double translationXStdDev;
  double translationYStdDev;
  double translationZStdDev;
};

}  // namespace CalibrationPresetsMaker
}  // namespace VideoStitch
