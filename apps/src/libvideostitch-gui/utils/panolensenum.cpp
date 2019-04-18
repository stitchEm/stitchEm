// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panolensenum.hpp"
#include <QApplication>

void PanoLensClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[VideoStitch::Core::PanoProjection::Type::Rectilinear] =
      QApplication::translate("PanoLens", "Rectilinear");
  enumToString[VideoStitch::Core::PanoProjection::Type::Cylindrical] =
      QApplication::translate("PanoLens", "Circular fisheye");
  enumToString[VideoStitch::Core::PanoProjection::Type::Equirectangular] =
      QApplication::translate("PanoLens", "Equirectangular");
  enumToString[VideoStitch::Core::PanoProjection::Type::FullFrameFisheye] =
      QApplication::translate("PanoLens", "Fullframe fisheye");
  enumToString[VideoStitch::Core::PanoProjection::Type::Stereographic] =
      QApplication::translate("PanoLens", "Stereographic");
  enumToString[VideoStitch::Core::PanoProjection::Type::CircularFisheye] =
      QApplication::translate("PanoLens", "Circular fisheye");
}

const PanoLensClass::Enum PanoLensClass::defaultValue = VideoStitch::Core::PanoProjection::Type::Rectilinear;
