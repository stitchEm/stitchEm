// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "displaymode.hpp"

#include <QRegExp>

QString displayModeToString(const VideoStitch::Plugin::DisplayMode& mode) {
  return QString::number(mode.width) + "x" + QString::number(mode.height) + " " +
         QString::number(mode.interleaved ? (mode.framerate.num / (double)mode.framerate.den) * 2
                                          : (mode.framerate.num / (double)mode.framerate.den),
                         'G', 4) +
         "fps" + QString(mode.interleaved ? " (interlaced)" : QString()) + QString(mode.psf ? " (PSF)" : QString());
}

bool lessThan(const VideoStitch::Plugin::DisplayMode& lhs, const VideoStitch::Plugin::DisplayMode& rhs) {
  if (lhs.width != rhs.width) {
    return lhs.width < rhs.width;
  }
  if (lhs.height != rhs.height) {
    return lhs.height < rhs.height;
  }
  int lFactor = 1;
  if (lhs.interleaved) {
    lFactor = 2;
  }
  int rFactor = 1;
  if (rhs.interleaved) {
    rFactor = 2;
  }
  if (lFactor * lhs.framerate.num != rFactor * rhs.framerate.num || lhs.framerate.den != rhs.framerate.den) {
    return lFactor * lhs.framerate.num * rhs.framerate.den < rFactor * rhs.framerate.num * lhs.framerate.den;
  }
  return lhs.interleaved != rhs.interleaved && lhs.interleaved == true;
}
