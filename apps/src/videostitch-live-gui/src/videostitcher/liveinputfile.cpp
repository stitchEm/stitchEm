// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputfile.hpp"

#include "libvideostitch/ptv.hpp"

LiveInputFile::LiveInputFile(const QString &name) : LiveInputFactory(name), width(1280), height(960), hasAudio(false) {}

LiveInputFile::~LiveInputFile() {}

VideoStitch::Ptv::Value *LiveInputFile::serialize() const {
  VideoStitch::Ptv::Value *input = VideoStitch::Ptv::Value::emptyObject();
  input->get("width")->asInt() = width;
  input->get("height")->asInt() = height;
  input->get("audio_enabled")->asBool() = hasAudio;
  input->get("reader_config")->asString() = name.toStdString();
  return input;
}

void LiveInputFile::initializeWith(const VideoStitch::Ptv::Value *initializationInput) {
  LiveInputFactory::initializeWith(initializationInput);
  width = initializationInput->has("width")->asInt();
  height = initializationInput->has("height")->asInt();
  if (initializationInput->has("audio_enabled")) {
    hasAudio = initializationInput->has("audio_enabled")->asBool();
  }
  name = QString::fromStdString(initializationInput->has("reader_config")->asString());
}

VideoStitch::InputFormat::InputFormatEnum LiveInputFile::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::MEDIA;
}

int LiveInputFile::getWidth() const { return width; }

int LiveInputFile::getHeight() const { return height; }

bool LiveInputFile::getHasAudio() const { return hasAudio; }

void LiveInputFile::setWidth(int newWidth) { width = newWidth; }

void LiveInputFile::setHeight(int newHeight) { height = newHeight; }

void LiveInputFile::setHasAudio(bool audio) { hasAudio = audio; }
