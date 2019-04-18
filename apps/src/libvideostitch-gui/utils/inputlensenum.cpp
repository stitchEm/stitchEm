// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputlensenum.hpp"

#include "libvideostitch/logging.hpp"

#include <QApplication>

#include <cassert>
#include <sstream>

static const float MAX_FOV_RECT(179.9f);
static const float MAX_FOV_DEF(359.9f);

void InputLensClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  // Only expose the legacy lens types to avoid confusing users
  enumToString[Enum::Rectilinear] = QApplication::translate("InputLens", "Rectilinear");
  enumToString[Enum::CircularFisheye] = QApplication::translate("InputLens", "Circular fisheye");
  enumToString[Enum::FullFrameFisheye] = QApplication::translate("InputLens", "Fullframe fisheye");
  enumToString[Enum::Equirectangular] = QApplication::translate("InputLens", "Equirectangular");
}

float InputLensClass::getMaximumFOV(const InputLensClass::Enum value) {
  if (value == Enum::Rectilinear)
    return MAX_FOV_RECT;
  else
    return MAX_FOV_DEF;
}

const InputLensClass::Enum InputLensClass::defaultValue = Enum::Rectilinear;

InputLensClass::LensType InputLensClass::getLensTypeFromInputDefinitionFormat(
    const VideoStitch::Core::InputDefinition::Format format) {
  // Hide the "optimized" lens types to the UI
  switch (format) {
    case VideoStitch::Core::InputDefinition::Format::FullFrameFisheye:
    case VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt:
      return LensType::FullFrameFisheye;

    case VideoStitch::Core::InputDefinition::Format::CircularFisheye:
    case VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt:
      return LensType::CircularFisheye;

    case VideoStitch::Core::InputDefinition::Format::Rectilinear:
      return LensType::Rectilinear;

    case VideoStitch::Core::InputDefinition::Format::Equirectangular:
      return LensType::Equirectangular;
  }

  assert(false);
  std::stringstream message;
  message << "Warning: unsupported input definition format " << static_cast<int>(format) << std::endl;
  VideoStitch::Logger::get(VideoStitch::Logger::Warning) << message.str();
  return LensType::FullFrameFisheye;
}

VideoStitch::Core::InputDefinition::Format InputLensClass::getInputDefinitionFormatFromLensType(
    const LensType lensType, const VideoStitch::Core::InputDefinition::LensModelCategory lensModelCategory) {
  // Translate the "legacy" lens types into optimized ones, if needed
  switch (lensType) {
    case LensType::Rectilinear:
      return VideoStitch::Core::InputDefinition::Format::Rectilinear;

    case LensType::FullFrameFisheye:
      if (lensModelCategory == VideoStitch::Core::InputDefinition::LensModelCategory::Optimized) {
        return VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt;
      } else {
        return VideoStitch::Core::InputDefinition::Format::FullFrameFisheye;
      }

    case LensType::CircularFisheye:
      if (lensModelCategory == VideoStitch::Core::InputDefinition::LensModelCategory::Optimized) {
        return VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt;
      } else {
        return VideoStitch::Core::InputDefinition::Format::CircularFisheye;
      }

    case LensType::Equirectangular:
      return VideoStitch::Core::InputDefinition::Format::Equirectangular;
  }

  assert(false);
  std::stringstream message;
  message << "Warning: unsupported lens type " << static_cast<int>(lensType) << std::endl;
  VideoStitch::Logger::get(VideoStitch::Logger::Warning) << message.str();
  return VideoStitch::Core::InputDefinition::Format::FullFrameFisheye;
}
