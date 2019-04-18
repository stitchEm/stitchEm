// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mpeg4codec.hpp"

Mpeg4Codec::Mpeg4Codec(QWidget *const parent) : MpegLikeCodec(parent) {}

void Mpeg4Codec::setup() {
  BasicMpegCodec::setup();
  addGOPConfiguration();
  showAdvancedConfiguration(false);
}

QString Mpeg4Codec::getKey() const { return QStringLiteral("mpeg4"); }

bool Mpeg4Codec::meetsSizeRequirements(int width, int height) const { return (width % 8 == 0 && height % 8 == 0); }

void Mpeg4Codec::correctSizeToMeetRequirements(int &width, int &height) {
  if (!meetsSizeRequirements(width, height)) {
    int hRest, wRest;
    hRest = height % 8;
    wRest = width % 8;
    int diff = wRest - (hRest * 2) % 8;
    // if width <16, height will be < 16 and height won't be a multiple of 8, thus it will be 16x8
    if (diff < 0 && width >= 16 && wRest != 0) {
      width = width - wRest;
      height = width / 2;
    } else {
      height = std::max(height, 8);
      height = height - hRest;
      width = height * 2;
    }
  }
}
