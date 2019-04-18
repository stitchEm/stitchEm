// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "smartenum.hpp"

namespace H264Config {

enum Preset { PLACEBO, VERYSLOW, SLOWER, SLOW, MEDIUM, FAST, FASTER, VERYFAST, SUPERFAST, ULTRAFAST };

enum Tune { FILM, ANIMATION, GRAIN, STILLIMAGE, PSNR, SSIM, FASTCODE, ZEROLATENCY };

enum Profile { BASELINE, MAIN, HIGH, HIGH10, HIGH422, HIGH444 };

class VS_GUI_EXPORT PresetClass {
 public:
  typedef Preset Enum;

  static void initDescriptions(QMap<Enum, QString>& enumToString);

  static const Enum defaultValue;
};

typedef SmartEnum<PresetClass, QString> PresetEnum;

class VS_GUI_EXPORT TuneClass {
 public:
  typedef Tune Enum;

  static void initDescriptions(QMap<Enum, QString>& enumToString);

  static const Enum defaultValue;
};

typedef SmartEnum<TuneClass, QString> TuneEnum;

class VS_GUI_EXPORT ProfileClass {
 public:
  typedef Profile Enum;

  static void initDescriptions(QMap<Enum, QString>& enumToString);

  static const Enum defaultValue;
};

typedef SmartEnum<ProfileClass, QString> ProfileEnum;

}  // namespace H264Config
