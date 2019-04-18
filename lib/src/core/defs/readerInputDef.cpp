// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"
#include "libvideostitch/readerInputDef.hpp"
#include "libvideostitch/logging.hpp"
#include "parse/json.hpp"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

namespace VideoStitch {
namespace Core {

ReaderInputDefinition::Pimpl::Pimpl()
    : readerConfig(Ptv::Value::emptyObject()),
      width(0),
      height(0),
      frameOffset(0),
      isVideoEnabled(true),
      isAudioEnabled(false) {}

ReaderInputDefinition::Pimpl::~Pimpl() { delete readerConfig; }

void ReaderInputDefinition::cloneTo(ReaderInputDefinition* dstDef) const {
#define PIMPL_FIELD_COPY(field) dstDef->pimpl->field = pimpl->field;
  PIMPL_FIELD_COPY(width);
  PIMPL_FIELD_COPY(height);
  PIMPL_FIELD_COPY(frameOffset);
  delete dstDef->pimpl->readerConfig;
  dstDef->pimpl->readerConfig = pimpl->readerConfig ? pimpl->readerConfig->clone() : Ptv::Value::emptyObject();
  PIMPL_FIELD_COPY(isVideoEnabled);
  PIMPL_FIELD_COPY(isAudioEnabled);
#undef PIMPL_FIELD_COPY
}

ReaderInputDefinition* ReaderInputDefinition::clone() const {
  ReaderInputDefinition* result = new ReaderInputDefinition();
  cloneTo(result);
  return result;
}

bool ReaderInputDefinition::operator==(const ReaderInputDefinition& other) const {
#define FIELD_EQUAL(getter) (getter() == other.getter())
  if (!(FIELD_EQUAL(getReaderConfig) && FIELD_EQUAL(getWidth) && FIELD_EQUAL(getHeight) &&
        FIELD_EQUAL(getFrameOffset))) {
    return false;
  }

  return true;
#undef FIELD_EQUAL
}

ReaderInputDefinition::ReaderInputDefinition() : pimpl(new Pimpl()) {}

ReaderInputDefinition::~ReaderInputDefinition() { delete pimpl; }

bool ReaderInputDefinition::validate(std::ostream& os) const {
  if (getWidth() <= 0) {
    os << "width must be strictly positive." << std::endl;
    return false;
  }
  if (getHeight() <= 0) {
    os << "height must be strictly positive." << std::endl;
    return false;
  }
  if (!pimpl->readerConfig) {
    os << "Missing reader config." << std::endl;
    return false;
  }

  return true;
}

Ptv::Value* Core::ReaderInputDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("reader_config", getReaderConfig().clone());

  res->push("width", new Parse::JsonValue((int64_t)getWidth()));
  res->push("height", new Parse::JsonValue((int64_t)getHeight()));
  res->push("frame_offset", new Parse::JsonValue(getFrameOffset()));

  // Inputs:
  res->push("video_enabled", new Parse::JsonValue(getIsVideoEnabled()));
  res->push("audio_enabled", new Parse::JsonValue(getIsAudioEnabled()));
  return res;
}

Core::ReaderInputDefinition* Core::ReaderInputDefinition::create(const Ptv::Value& value, bool enforceMandatoryFields) {
  std::unique_ptr<ReaderInputDefinition> res(new ReaderInputDefinition());
  if (!res->applyDiff(value, enforceMandatoryFields).ok()) {
    return nullptr;
  }
  return res.release();
}

Status Core::ReaderInputDefinition::applyDiff(const Ptv::Value& value, bool enforceMandatoryFields) {
  // Make sure value is an object.
  if (!Parse::checkType("ReaderInputDefinition", value, Ptv::Value::OBJECT)) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
            "Could not find valid 'ReaderInputDefinition', expected object type"};
  }
  // Support for old files:
  const Ptv::Value* readerConfig = value.has("reader_config") ? value.has("reader_config") : value.has("filename");
  if (!readerConfig) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Encountered invalid input definition"};
  }
  setReaderConfig(readerConfig->clone());

  // TODOLATERSTATUS: this should be handled in the populate function, not here through macros
#define POPULATE_INT_PROPAGATE_WRONGTYPE(config_name, varName, shouldEnforce)                 \
  if (Parse::populateInt("InputDefinition", value, config_name, varName, shouldEnforce) ==    \
      Parse::PopulateResult_WrongType) {                                                      \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                     \
            "Invalid type for '" config_name "' in InputDefinition, expected integer value"}; \
  }
#define POPULATE_BOOL_PROPAGATE_WRONGTYPE(config_name, varName, shouldEnforce)                \
  if (Parse::populateBool("InputDefinition", value, config_name, varName, shouldEnforce) ==   \
      Parse::PopulateResult_WrongType) {                                                      \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                     \
            "Invalid type for '" config_name "' in InputDefinition, expected boolean value"}; \
  }

  POPULATE_BOOL_PROPAGATE_WRONGTYPE("video_enabled", pimpl->isVideoEnabled, false);
  POPULATE_BOOL_PROPAGATE_WRONGTYPE("audio_enabled", pimpl->isAudioEnabled, false);
  // backward-compat
  // TODO FIXMELATER not sure this it the proper way to go for audio-only inputs
  POPULATE_BOOL_PROPAGATE_WRONGTYPE("enabled", pimpl->isVideoEnabled, false);
  // Support for audio-only inputs
  if (pimpl->isAudioEnabled && !pimpl->isVideoEnabled) {
    return Status::OK();
  }

  POPULATE_INT_PROPAGATE_WRONGTYPE("width", pimpl->width, enforceMandatoryFields);
  POPULATE_INT_PROPAGATE_WRONGTYPE("height", pimpl->height, enforceMandatoryFields);
  if (enforceMandatoryFields) {
    POPULATE_INT_PROPAGATE_WRONGTYPE("frame_offset", pimpl->frameOffset, false);
  }

#undef POPULATE_INT_PROPAGATE_WRONGTYPE
#undef POPULATE_BOOL_PROPAGATE_WRONGTYPE

  return Status::OK();
}

const Ptv::Value& ReaderInputDefinition::getReaderConfig() const { return *pimpl->readerConfig; }

const Ptv::Value* ReaderInputDefinition::getReaderConfigPtr() const { return pimpl->readerConfig; }

void ReaderInputDefinition::setReaderConfig(Ptv::Value* config) {
  if (!config) {
    return;
  }
  delete pimpl->readerConfig;
  pimpl->readerConfig = config;
}

void ReaderInputDefinition::setFilename(const std::string& fileName) {
  if (!pimpl->readerConfig) {
    pimpl->readerConfig = Ptv::Value::emptyObject();
  }
  pimpl->readerConfig->asString() = fileName;
}

std::string ReaderInputDefinition::getDisplayName() const {
  // if the reader is file-based, return the file name
  if (pimpl->readerConfig->getType() == Ptv::Value::STRING) {
    return pimpl->readerConfig->asString();
  } else {
    std::string name;
    if (VideoStitch::Parse::populateString("ReaderInputDefinition", *pimpl->readerConfig, "name", name, false) !=
        VideoStitch::Parse::PopulateResult_Ok) {
      return std::string();
    }
    return name;
  }
  return std::string();
}

// Reseter for Geometries is defined explicitly with an argument, see resetGeometries(const double) below
GENGETSETTER(ReaderInputDefinition, int64_t, Width, width)
GENGETSETTER(ReaderInputDefinition, int64_t, Height, height)
GENGETSETTER(ReaderInputDefinition, int, FrameOffset, frameOffset)

void ReaderInputDefinition::setIsEnabled(bool) {
  // TODO FIXMELATER: have a proper implementation taking into account audio and video inputs
  // which can enable/disable getIsVideoEnabled() and getIsAudioEnabled() differently for audio and video inputs
  assert(false && "fix me");
}

bool ReaderInputDefinition::getIsEnabled() const {
  // TODO FIXMELATER have a proper implementation
  return true;
}

bool ReaderInputDefinition::getIsVideoEnabled() const { return pimpl->isVideoEnabled; }

void ReaderInputDefinition::setIsVideoEnabled(bool b) { pimpl->isVideoEnabled = b; }

bool ReaderInputDefinition::getIsAudioEnabled() const { return pimpl->isAudioEnabled; }

void ReaderInputDefinition::setIsAudioEnabled(bool b) { pimpl->isAudioEnabled = b; }

}  // namespace Core
}  // namespace VideoStitch
