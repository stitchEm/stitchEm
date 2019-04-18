// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputv4l2.hpp"

LiveInputV4L2::LiveInputV4L2(const QString &name) : CaptureCardLiveInput(name) {}

LiveInputV4L2::~LiveInputV4L2() {}

VideoStitch::InputFormat::InputFormatEnum LiveInputV4L2::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::V4L2;
}
