// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputprocedural.hpp"

#include "libvideostitch/ptv.hpp"

// -------------------------- Procedural namespace --------------------------

namespace Procedural {
void NameClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[frameNumber] = "frameNumber";
  enumToString[grid] = "grid";
  enumToString[color] = "color";
}

const NameClass::Enum NameClass::defaultValue = frameNumber;
}  // namespace Procedural

// -------------------------- LiveInputProcedural class --------------------------

LiveInputProcedural::LiveInputProcedural(const QString& name)
    : LiveInputFactory(name), width(1920), height(1080), color("#5520df") {}

LiveInputProcedural::~LiveInputProcedural() {}

VideoStitch::Ptv::Value* LiveInputProcedural::serialize() const {
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  value->get("type")->asString() = VideoStitch::InputFormat::getStringFromEnum(getType()).toStdString();
  value->get("name")->asString() = name.toStdString();
  value->get("color")->asString() = color.name().right(6).toStdString();

  VideoStitch::Ptv::Value* input = VideoStitch::Ptv::Value::emptyObject();
  input->get("width")->asInt() = width;
  input->get("height")->asInt() = height;
  input->push("reader_config", value);
  return input;
}

void LiveInputProcedural::initializeWith(const VideoStitch::Ptv::Value* initializationInput) {
  LiveInputFactory::initializeWith(initializationInput);

  width = initializationInput->has("width")->asInt();
  height = initializationInput->has("height")->asInt();
  QString colorString =
      QString("#") + QString::fromStdString(initializationInput->has("reader_config")->has("color")->asString());
  color.setNamedColor(colorString);
}

VideoStitch::InputFormat::InputFormatEnum LiveInputProcedural::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::PROCEDURAL;
}

int LiveInputProcedural::getWidth() const { return width; }

int LiveInputProcedural::getHeight() const { return height; }

QColor LiveInputProcedural::getColor() const { return color; }

void LiveInputProcedural::setWidth(int newWidth) { width = newWidth; }

void LiveInputProcedural::setHeight(int newHeight) { height = newHeight; }

void LiveInputProcedural::setColor(QColor newColor) { color = newColor; }
