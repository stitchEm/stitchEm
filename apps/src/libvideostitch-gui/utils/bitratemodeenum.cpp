// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "bitratemodeenum.hpp"

void BitRateModeClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[VBR] = "VBR";
  enumToString[CBR] = "CBR";
}

const BitRateModeClass::Enum BitRateModeClass::defaultValue = CUSTOM;
