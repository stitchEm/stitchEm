// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEINPUTXIMEA_H
#define LIVEINPUTXIMEA_H

#include "capturecardliveinput.hpp"

class LiveInputXimea : public CaptureCardLiveInput {
 public:
  explicit LiveInputXimea(const QString& name);
  ~LiveInputXimea();

  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;
  virtual VideoStitch::Ptv::Value* serialize() const;
};

#endif  // LIVEINPUTXIMEA_H
