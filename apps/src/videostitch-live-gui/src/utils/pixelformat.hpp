// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PIXELFORMATENUM_HPP
#define PIXELFORMATENUM_HPP

#include "libvideostitch-gui/utils/smartenum.hpp"
#include "libvideostitch/frame.hpp"

class PixelFormatClass {
 public:
  typedef VideoStitch::PixelFormat Enum;

  static void initDescriptions(QMap<Enum, QString>& enumToString);

  static const Enum defaultValue;
};

typedef SmartEnum<PixelFormatClass, QString> PixelFormatEnum;

#endif  // PIXELFORMATENUM_HPP
