// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveexposure.hpp"
#include "libvideostitch/parse.hpp"

LiveExposure::LiveExposure() : algorithm("exposure_stabilize"), anchor(-1), autoStart(false) {}

VideoStitch::Ptv::Value* LiveExposure::serialize() const {
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  value->get("algorithm")->asString() = algorithm.toStdString();
  value->get("anchor")->asInt() = anchor;
  value->get("auto")->asBool() = autoStart;
  return value;
}

const QString LiveExposure::getAlgorithm() const { return algorithm; }

int LiveExposure::getAnchor() const { return anchor; }

bool LiveExposure::getIsAutoStart() const { return autoStart; }

void LiveExposure::setAlgorithm(const QString& algo) { algorithm = algo; }

void LiveExposure::setAnchor(const int id) { anchor = id; }

void LiveExposure::setIsAutoStart(const bool isAuto) { autoStart = isAuto; }
