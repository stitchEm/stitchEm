// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputfactory.hpp"
#include "liveinputdecklink.hpp"
#include "liveinputfile.hpp"
#include "liveinputmagewell.hpp"
#include "liveinputximea.hpp"
#include "liveinputaja.hpp"
#include "liveinputprocedural.hpp"
#include "liveinputstream.hpp"
#include "liveinputv4l2.hpp"

#include "libvideostitch/ptv.hpp"

LiveInputFactory* LiveInputFactory::makeLiveInput(const VideoStitch::InputFormat::InputFormatEnum choice,
                                                  const QString& name) {
  switch (choice) {
    case VideoStitch::InputFormat::InputFormatEnum::PROCEDURAL:
      return new LiveInputProcedural(name);
    case VideoStitch::InputFormat::InputFormatEnum::MEDIA:
      return new LiveInputFile(name);
    case VideoStitch::InputFormat::InputFormatEnum::DECKLINK:
      return new LiveInputDecklink(name);
    case VideoStitch::InputFormat::InputFormatEnum::MAGEWELL:
      return new LiveInputMagewell(name);
    case VideoStitch::InputFormat::InputFormatEnum::MAGEWELLPRO:
      return new LiveInputMagewellPro(name);
    case VideoStitch::InputFormat::InputFormatEnum::AJA:
      return new LiveInputAJA(name);
    case VideoStitch::InputFormat::InputFormatEnum::XIMEA:
      return new LiveInputXimea(name);
    case VideoStitch::InputFormat::InputFormatEnum::V4L2:
      return new LiveInputV4L2(name);
    case VideoStitch::InputFormat::InputFormatEnum::NETWORK:
      return new LiveInputStream(name);
    default:
      return nullptr;
  }
}

LiveInputFactory* LiveInputFactory::makeLiveInput(const VideoStitch::InputFormat::InputFormatEnum choice,
                                                  const VideoStitch::Ptv::Value* initializationInput) {
  LiveInputFactory* liveInput = makeLiveInput(choice, QString());
  if (liveInput) {
    liveInput->initializeWith(initializationInput);
  }
  return liveInput;
}

LiveInputFactory::LiveInputFactory(const QString& name) : name(name) {}

LiveInputFactory::~LiveInputFactory() {}

void LiveInputFactory::initializeWith(const VideoStitch::Ptv::Value* initializationInput) {
  if (initializationInput->has("reader_config")->has("name")) {
    name = QString::fromStdString(initializationInput->has("reader_config")->has("name")->asString());
  } else {
    name = QString::fromStdString(initializationInput->has("reader_config")->asString());
  }
}

const QString LiveInputFactory::getName() const { return name; }

void LiveInputFactory::setName(QString newName) { name = newName; }
