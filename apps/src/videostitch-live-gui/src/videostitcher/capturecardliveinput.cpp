// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "capturecardliveinput.hpp"

#include "libvideostitch/ptv.hpp"

static const int DEFAULT_INPUT_WIDTH(1920);
static const int DEFAULT_INPUT_HEIGHT(1080);

static const VideoStitch::FrameRate getDefaultFrameRate() { return {25, 1}; }

CaptureCardLiveInput::CaptureCardLiveInput(const QString& name)
    : LiveInputFactory(name),
      displayMode(DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT, false, getDefaultFrameRate()),
      pixelFormat() {}

CaptureCardLiveInput::~CaptureCardLiveInput() {}

VideoStitch::Ptv::Value* CaptureCardLiveInput::serialize() const {
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  // Video setup
  value->get("name")->asString() = name.toStdString();
  value->get("type")->asString() = VideoStitch::InputFormat::getStringFromEnum(getType()).toStdString();
  value->get("pixel_format")->asString() = getPixelFormat().getDescriptor().toStdString();
  value->get("interleaved")->asBool() = displayMode.interleaved;
  VideoStitch::Ptv::Value* fps = VideoStitch::Ptv::Value::emptyObject();
  fps->get("num")->asInt() = displayMode.framerate.num;
  fps->get("den")->asInt() = displayMode.framerate.den;
  value->push("frame_rate", fps);
  // Reader setup
  VideoStitch::Ptv::Value* input = VideoStitch::Ptv::Value::emptyObject();
  input->get("width")->asInt() = displayMode.width;
  input->get("height")->asInt() = displayMode.height;
  input->push("reader_config", value);
  return input;
}

void CaptureCardLiveInput::initializeWith(const VideoStitch::Ptv::Value* initializationInput) {
  LiveInputFactory::initializeWith(initializationInput);

  const VideoStitch::Ptv::Value* reader_config = initializationInput->has("reader_config");
  const VideoStitch::Ptv::Value* interleavedValue = reader_config->has("interleaved");
  const VideoStitch::Ptv::Value* fpsValue = reader_config->has("frame_rate");
  const VideoStitch::Ptv::Value* pixelFormatValue = reader_config->has("pixel_format");

  displayMode.width = initializationInput->has("width")->asInt();
  displayMode.height = initializationInput->has("height")->asInt();
  displayMode.interleaved = interleavedValue ? interleavedValue->asBool() : false;
  displayMode.framerate = getDefaultFrameRate();
  if (fpsValue) {
    const VideoStitch::Ptv::Value* num = fpsValue->has("num");
    const VideoStitch::Ptv::Value* den = fpsValue->has("den");
    if (num && den) {
      displayMode.framerate.num = num->asInt();
      displayMode.framerate.den = den->asInt();
    }
  }

  const QString pixelformatStr = pixelFormatValue ? QString::fromStdString(pixelFormatValue->asString()) : QString();
  pixelFormat = PixelFormatEnum::getEnumFromDescriptor(pixelformatStr);
}

VideoStitch::Plugin::DisplayMode CaptureCardLiveInput::getDisplayMode() const { return displayMode; }

PixelFormatEnum CaptureCardLiveInput::getPixelFormat() const { return pixelFormat; }

void CaptureCardLiveInput::setDisplayMode(const VideoStitch::Plugin::DisplayMode& newDisplayMode) {
  displayMode = newDisplayMode;
}

void CaptureCardLiveInput::setPixelFormat(const PixelFormatEnum& newPixelFormat) { pixelFormat = newPixelFormat; }
