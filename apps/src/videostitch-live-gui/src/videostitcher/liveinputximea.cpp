// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputximea.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"

LiveInputXimea::LiveInputXimea(const QString &name) : CaptureCardLiveInput(name) {
  // Default values for DeckLink
  displayMode = VideoStitch::Plugin::DisplayMode(1920, 1080, false, {30, 1});
  pixelFormat = PixelFormatEnum(VideoStitch::UYVY);
}

LiveInputXimea::~LiveInputXimea() {}

VideoStitch::InputFormat::InputFormatEnum LiveInputXimea::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::XIMEA;
}

VideoStitch::Ptv::Value *LiveInputXimea::serialize() const {
  VideoStitch::Logger::get(VideoStitch::Logger::Error)
      << "[Ximea] Ximea live input " << name.toStdString() << " : " << name.right(1).toInt() << std::endl;

  VideoStitch::Ptv::Value *input = CaptureCardLiveInput::serialize();
  input->get("reader_config")->get("device")->asInt() = name.right(1).toInt();
  return input;
}
