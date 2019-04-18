// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationPresetsMakerConfig.hpp"

#include "libvideostitch/parse.hpp"

#include <memory>

namespace VideoStitch {
namespace CalibrationPresetsMaker {

CalibrationPresetsMakerConfig::CalibrationPresetsMakerConfig(const Ptv::Value* config)
    : isConfigValid(true),
      presetsName(""),
      focalStdDevValuePercentage(PTV_DEFAULT_INPUTDEF_TEMPLATE_FOCAL_STD_DEV_VALUE_PERCENTAGE),
      centerStdDevWidthPercentage(PTV_DEFAULT_INPUTDEF_TEMPLATE_CENTER_STD_DEV_WIDTH_PERCENTAGE),
      distortStdDevValuePercentage(PTV_DEFAULT_INPUTDEF_TEMPLATE_DISTORT_STD_DEV_VALUE_PERCENTAGE),
      yawStdDev(PTV_DEFAULT_INPUTDEF_TEMPLATE_ANGLE_STD_DEV),
      pitchStdDev(PTV_DEFAULT_INPUTDEF_TEMPLATE_ANGLE_STD_DEV),
      rollStdDev(PTV_DEFAULT_INPUTDEF_TEMPLATE_ANGLE_STD_DEV),
      translationXStdDev(PTV_DEFAULT_INPUTDEF_TEMPLATE_TRANSLATION_STD_DEV),
      translationYStdDev(PTV_DEFAULT_INPUTDEF_TEMPLATE_TRANSLATION_STD_DEV),
      translationZStdDev(PTV_DEFAULT_INPUTDEF_TEMPLATE_TRANSLATION_STD_DEV) {
  if (!config) {
    isConfigValid = false;
    return;
  }

  if (Parse::populateString("presets_name", *config, "name", presetsName, true) != Parse::PopulateResult::OK) {
    isConfigValid = false;
    return;
  }

  auto populateOptionalDoubleValue = [&](double& value, const std::string& name) {
    if (VideoStitch::Parse::populateDouble(name, *config, name, value, false) == Parse::PopulateResult::WrongType) {
      isConfigValid = false;
    }
  };

  populateOptionalDoubleValue(focalStdDevValuePercentage, "focal_std_dev_value_percentage");
  populateOptionalDoubleValue(centerStdDevWidthPercentage, "center_std_dev_width_percentage");
  populateOptionalDoubleValue(distortStdDevValuePercentage, "distort_std_dev_value_percentage");
  populateOptionalDoubleValue(yawStdDev, "yaw_std_dev");
  populateOptionalDoubleValue(pitchStdDev, "pitch_std_dev");
  populateOptionalDoubleValue(rollStdDev, "roll_std_dev");
  populateOptionalDoubleValue(translationXStdDev, "translation_x_std_dev");
  populateOptionalDoubleValue(translationYStdDev, "translation_y_std_dev");
  populateOptionalDoubleValue(translationZStdDev, "translation_z_std_dev");
}

CalibrationPresetsMakerConfig::CalibrationPresetsMakerConfig(const CalibrationPresetsMakerConfig& other)
    : isConfigValid(other.isConfigValid),
      presetsName(other.presetsName),
      focalStdDevValuePercentage(other.focalStdDevValuePercentage),
      centerStdDevWidthPercentage(other.centerStdDevWidthPercentage),
      distortStdDevValuePercentage(other.distortStdDevValuePercentage),
      yawStdDev(other.yawStdDev),
      pitchStdDev(other.pitchStdDev),
      rollStdDev(other.rollStdDev),
      translationXStdDev(other.translationXStdDev),
      translationYStdDev(other.translationYStdDev),
      translationZStdDev(other.translationZStdDev) {}

}  // namespace CalibrationPresetsMaker
}  // namespace VideoStitch
