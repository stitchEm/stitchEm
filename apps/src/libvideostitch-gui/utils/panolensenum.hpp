// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PANOLENSENUM_HPP
#define PANOLENSENUM_HPP

#include "smartenum.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/projections.hpp"

class VS_GUI_EXPORT PanoLensClass {
 public:
  typedef VideoStitch::Core::PanoProjection::Type Enum;

  static void initDescriptions(QMap<Enum, QString>& enumToString);

  static const Enum defaultValue;
};

typedef SmartEnum<PanoLensClass, QString> PanoLensEnum;

#endif  // PANOLENSENUM_HPP
