// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common-config.hpp"

#include <QString>
#include <QMap>

namespace VideoStitch {

enum Projection {
  unknownProjection = -1,
  minView = 0,
  rectilinear = 0,
  equirectangular = 1,
  fullframe_fisheye = 2,
  circular_fisheye = 3,
  stereographic = 4,
  interactive = 5,  // ex-spherical
  cubemap = 6,
  equiangular_cubemap = 7,
  maxView = interactive
};

/**
 * @brief Maps a QString dedicated to be displayed in the GUI to an index
 */
QMap<QString, Projection> VS_COMMON_EXPORT initMapGUIStringToIndex();
QMap<QString, Projection> VS_COMMON_EXPORT initMapPTVStringToIndex();

const QMap<QString, Projection> mapGUIStringToIndex = initMapGUIStringToIndex();
const QMap<QString, Projection> mapPTVStringToIndex = initMapPTVStringToIndex();

/**
 * QStringList containing the available projection in VideoStitch
 */
QStringList VS_COMMON_EXPORT guiStringListProjection();

double VS_COMMON_EXPORT getMinimumValueFor(Projection projection);
double VS_COMMON_EXPORT getMaximumValueFor(Projection projection);
double VS_COMMON_EXPORT getDefaultFor(Projection projection);
}  // namespace VideoStitch
