// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputstream.hpp"
#include "libvideostitch/ptv.hpp"

static const int DEFAULT_INPUT_WIDTH(1920);
static const int DEFAULT_INPUT_HEIGHT(1080);

LiveInputStream::LiveInputStream(const QString& name)
    : LiveInputFactory(name), width(DEFAULT_INPUT_WIDTH), height(DEFAULT_INPUT_HEIGHT) {}

VideoStitch::Ptv::Value* LiveInputStream::serialize() const {
  VideoStitch::Ptv::Value* input = VideoStitch::Ptv::Value::emptyObject();
  input->get("reader_config")->asString() = name.toStdString();
  input->get("width")->asInt() = width;
  input->get("height")->asInt() = height;
  return input;
}

void LiveInputStream::initializeWith(const VideoStitch::Ptv::Value* initializationInput) {
  width = initializationInput->has("width")->asInt();
  height = initializationInput->has("height")->asInt();
}

VideoStitch::InputFormat::InputFormatEnum LiveInputStream::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::NETWORK;
}
