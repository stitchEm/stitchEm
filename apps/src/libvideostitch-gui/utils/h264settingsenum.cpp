// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "h264settingsenum.hpp"
namespace H264Config {

void PresetClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[PLACEBO] = "placebo";
  enumToString[VERYSLOW] = "veryslow";
  enumToString[SLOWER] = "slower";
  enumToString[SLOW] = "slow";
  enumToString[MEDIUM] = "medium";
  enumToString[FAST] = "fast";
  enumToString[FASTER] = "faster";
  enumToString[VERYFAST] = "veryfast";
  enumToString[SUPERFAST] = "superfast";
  enumToString[ULTRAFAST] = "ultrafast";
}

const PresetClass::Enum PresetClass::defaultValue = MEDIUM;

void TuneClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[FILM] = "film";
  enumToString[ANIMATION] = "animation";
  enumToString[GRAIN] = "grain";
  enumToString[STILLIMAGE] = "stillimage";
  enumToString[PSNR] = "psnr";
  enumToString[SSIM] = "ssim";
  enumToString[FASTCODE] = "fastdecode";
  enumToString[ZEROLATENCY] = "zerolatency";
}

const TuneClass::Enum TuneClass::defaultValue = ZEROLATENCY;

void ProfileClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[BASELINE] = "baseline";
  enumToString[MAIN] = "main";
  enumToString[HIGH] = "high";
  enumToString[HIGH10] = "high10";
  enumToString[HIGH422] = "high422";
  enumToString[HIGH444] = "high444";
}

const ProfileClass::Enum ProfileClass::defaultValue = BASELINE;

}  // namespace H264Config
