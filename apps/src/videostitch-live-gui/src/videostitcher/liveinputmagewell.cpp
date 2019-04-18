// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputmagewell.hpp"

#include "libvideostitch/ptv.hpp"

// -------------------------- Magewell namespace --------------------------

namespace Magewell {
void BuiltInZoomClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[Zoom] = "zoom";
  enumToString[Fill] = "fill";
  enumToString[None] = "none";
}

const BuiltInZoomClass::Enum BuiltInZoomClass::defaultValue = None;
}  // namespace Magewell

// -------------------------- LiveInputMagewell class --------------------------

LiveInputMagewell::LiveInputMagewell(const QString& name) : CaptureCardLiveInput(name), builtInZoom() {
  // Default values for Magewell
  displayMode = VideoStitch::Plugin::DisplayMode(1920, 1080, false, {30, 1});
  pixelFormat = PixelFormatEnum(VideoStitch::PixelFormat::YUY2);
}

LiveInputMagewell::~LiveInputMagewell() {}

VideoStitch::Ptv::Value* LiveInputMagewell::serialize() const {
  VideoStitch::Ptv::Value* input = CaptureCardLiveInput::serialize();
  input->get("reader_config")->get("builtin_zoom")->asString() = builtInZoom.getDescriptor().toStdString();
  return input;
}

void LiveInputMagewell::initializeWith(const VideoStitch::Ptv::Value* initializationInput) {
  CaptureCardLiveInput::initializeWith(initializationInput);
  const QString zoomString =
      QString::fromStdString(initializationInput->has("reader_config")->has("builtin_zoom")->asString());
  builtInZoom = Magewell::BuiltInZoomEnum::getEnumFromDescriptor(zoomString);
}

VideoStitch::InputFormat::InputFormatEnum LiveInputMagewell::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::MAGEWELL;
}

Magewell::BuiltInZoomEnum LiveInputMagewell::getBuiltInZoom() const { return builtInZoom; }

void LiveInputMagewell::setBuildInZoom(const Magewell::BuiltInZoomEnum& newBuiltInZoom) {
  builtInZoom = newBuiltInZoom;
}

LiveInputMagewellPro::LiveInputMagewellPro(const QString& name) : LiveInputMagewell(name) {}

LiveInputMagewellPro::~LiveInputMagewellPro() {}

VideoStitch::InputFormat::InputFormatEnum LiveInputMagewellPro::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::MAGEWELLPRO;
}
