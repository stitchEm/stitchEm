// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef BITRATEMODEENUM_HPP
#define BITRATEMODEENUM_HPP

#include "smartenum.hpp"

enum BitRateMode { VBR, CBR, CUSTOM };

class VS_GUI_EXPORT BitRateModeClass {
 public:
  typedef BitRateMode Enum;

  static void initDescriptions(QMap<Enum, QString>& enumToString);

  static const Enum defaultValue;
};

typedef SmartEnum<BitRateModeClass, QString> BitRateModeEnum;

#endif  // BITRATEMODEENUM_HPP
