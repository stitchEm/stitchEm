// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEINPUTDECKLINK_H
#define LIVEINPUTDECKLINK_H

#include "capturecardliveinput.hpp"

class LiveInputDecklink : public CaptureCardLiveInput {
 public:
  explicit LiveInputDecklink(const QString &name);
  ~LiveInputDecklink();

  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;
};

#endif  // LIVEINPUTDECKLINK_H
