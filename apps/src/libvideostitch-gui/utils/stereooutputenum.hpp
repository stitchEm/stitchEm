// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef STEREOOUTPUTENUM_HPP
#define STEREOOUTPUTENUM_HPP

#include "smartenum.hpp"

enum StereoOutputType { PANORAMA, LEFT_EYE, RIGHT_EYE, SIDE_BY_SIDE, ABOVE_BELOW, ANAGLYPH, OCULUS };

class VS_GUI_EXPORT StereoOutputClass {
 public:
  typedef StereoOutputType Enum;

  static void initDescriptions(QMap<Enum, QString>& enumToString);

  static const Enum defaultValue;
};

typedef SmartEnum<StereoOutputClass, QString> StereoOutputEnum;

#endif  // STEREOOUTPUTENUM_HPP
