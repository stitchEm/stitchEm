// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/stitchOutput.hpp"

namespace VideoStitch {
namespace Output {
class DiscardVideoWriter : public VideoWriter {
 public:
  DiscardVideoWriter(const std::string& name, unsigned width, unsigned height, FrameRate framerate)
      : Output(name), VideoWriter(width, height, framerate, VideoStitch::PixelFormat::RGBA) {}

  void pushVideo(const Frame&) {}
};
}  // namespace Output
}  // namespace VideoStitch
