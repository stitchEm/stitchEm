// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "capturecardliveinput.hpp"

class LiveInputV4L2 : public CaptureCardLiveInput {
 public:
  explicit LiveInputV4L2(const QString &name);
  ~LiveInputV4L2();

  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;
};
