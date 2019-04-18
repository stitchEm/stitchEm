// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputdecklink.hpp"

LiveInputDecklink::LiveInputDecklink(const QString &name) : CaptureCardLiveInput(name) {
  // Default values for DeckLink
  displayMode = VideoStitch::Plugin::DisplayMode(1920, 1080, false, {30, 1});
  pixelFormat = PixelFormatEnum(VideoStitch::UYVY);
}

LiveInputDecklink::~LiveInputDecklink() {}

VideoStitch::InputFormat::InputFormatEnum LiveInputDecklink::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::DECKLINK;
}
