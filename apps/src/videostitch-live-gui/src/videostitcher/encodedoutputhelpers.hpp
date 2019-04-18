// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/utils/videocodecs.hpp"

namespace VideoStitch {

int getDefaultBitrate(double pixelRate);  // In kbit/s
QString getLevelFromMacroblocksRate(int macroblocksRate, VideoCodec::VideoCodecEnum codec);
int getMaxConstantBitRate(QString profile, QString level, VideoCodec::VideoCodecEnum codec);  // In  kbits/s
int getMacroblocksRate(double pixelRate);
double getPixelRate(int width, int height);

}  // namespace VideoStitch
