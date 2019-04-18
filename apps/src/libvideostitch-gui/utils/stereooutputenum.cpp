// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stereooutputenum.hpp"
#include <QApplication>

void StereoOutputClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[PANORAMA] = QApplication::translate("StereoOutput", "Panorama");
  enumToString[LEFT_EYE] = QApplication::translate("StereoOutput", "Left Eye");
  enumToString[RIGHT_EYE] = QApplication::translate("StereoOutput", "Right Eye");
  enumToString[SIDE_BY_SIDE] = QApplication::translate("StereoOutput", "Side-By-Side");
  enumToString[ABOVE_BELOW] = QApplication::translate("StereoOutput", "Above-Below");
  enumToString[ANAGLYPH] = QApplication::translate("StereoOutput", "Anaglyph");
  enumToString[OCULUS] = QApplication::translate("StereoOutput", "Oculus Rift");
}

const StereoOutputClass::Enum StereoOutputClass::defaultValue = PANORAMA;
