// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef INPUTLENSENUM_HPP
#define INPUTLENSENUM_HPP

#include "smartenum.hpp"

#include "libvideostitch/inputDef.hpp"

class VS_GUI_EXPORT InputLensClass {
 public:
  typedef enum class LensType { Rectilinear, CircularFisheye, FullFrameFisheye, Equirectangular } Enum;

  static void initDescriptions(QMap<Enum, QString>& enumToString);

  static const Enum defaultValue;

  static float getMaximumFOV(const Enum value);

  /**
   * @brief Translate InputDefinition::Format into LensType
   * @param format the InputDefinition format
   * @return The lens type (Rectilinear, FullFrameFisheye, CircularFisheye, Equirectangular)
   * @note This is meant to hide the "optimized" lens models to the users to avoid confusing them
   */
  static LensType getLensTypeFromInputDefinitionFormat(const VideoStitch::Core::InputDefinition::Format format);

  /**
   * @brief Translate LensType into InputDefinition::Format
   * @param lensType the UI lens type
   * @param lensModelCategory the lens model category (related to inputs)
   * @return InputDefinition::Format lens format
   * @note This is meant to hide the "optimized" lens models to the users to avoid confusing them
   */
  static VideoStitch::Core::InputDefinition::Format getInputDefinitionFormatFromLensType(
      const LensType lensType, const VideoStitch::Core::InputDefinition::LensModelCategory lensModelCategory);
};

typedef SmartEnum<InputLensClass, QString> InputLensEnum;

#endif  // INPUTLENSENUM_HPP
