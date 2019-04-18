// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "projection.hpp"

#include <QStringList>

namespace VideoStitch {

QMap<QString, Projection> initMapGUIStringToIndex() {
  QMap<QString, Projection> string2ProjectionMap;
  string2ProjectionMap[QT_TR_NOOP("Rectilinear")] = rectilinear;
  string2ProjectionMap[QT_TR_NOOP("Equirectangular")] = equirectangular;
  string2ProjectionMap[QT_TR_NOOP("Fullframe fisheye")] = fullframe_fisheye;
  string2ProjectionMap[QT_TR_NOOP("Circular fisheye")] = circular_fisheye;
  string2ProjectionMap[QT_TR_NOOP("Stereographic")] = stereographic;
  string2ProjectionMap[QT_TR_NOOP("Cubemap")] = cubemap;
  string2ProjectionMap[QT_TR_NOOP("Equiangular cubemap")] = equiangular_cubemap;
  return string2ProjectionMap;
}

QMap<QString, Projection> initMapPTVStringToIndex() {
  QMap<QString, Projection> string2ProjectionMap;
  string2ProjectionMap["rectilinear"] = rectilinear;
  string2ProjectionMap["equirectangular"] = equirectangular;
  string2ProjectionMap["ff_fisheye"] = fullframe_fisheye;
  string2ProjectionMap["circular_fisheye"] = circular_fisheye;
  string2ProjectionMap["stereographic"] = stereographic;
  string2ProjectionMap["cubemap"] = cubemap;
  string2ProjectionMap["equiangular_cubemap"] = equiangular_cubemap;
  return string2ProjectionMap;
}

QStringList guiStringListProjection() {
  // Keep the order of the enum int value
  return QStringList() << QT_TR_NOOP("Rectilinear") << QT_TR_NOOP("Equirectangular") << QT_TR_NOOP("Fullframe fisheye")
                       << QT_TR_NOOP("Circular fisheye") << QT_TR_NOOP("Stereographic") << QT_TR_NOOP("Cubemap")
                       << QT_TR_NOOP("Equiangular cubemap");
}

double getMinimumValueFor(Projection projection) {
  Q_UNUSED(projection);
  return 0.0;
}

double getMaximumValueFor(Projection projection) {
  switch (projection) {
    case rectilinear:
      return 179.9;
    case cubemap:
    case equiangular_cubemap:
    case equirectangular:
    case fullframe_fisheye:
    case circular_fisheye:
      return 360.0;
    case stereographic:
      return 359.9;
    default:
      return 0.0;
  }
}

double getDefaultFor(Projection projection) {
  switch (projection) {
    case rectilinear:
      return 160.0;
    case equirectangular:
    case cubemap:
    case equiangular_cubemap:
      return 360.0;
    case fullframe_fisheye:
    case circular_fisheye:
    case stereographic:
      return 320.0;
    default:
      return 0.0;
  }
}

}  // namespace VideoStitch
