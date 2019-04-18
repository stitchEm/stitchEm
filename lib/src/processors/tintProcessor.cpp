// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "tintProcessor.hpp"

#include "backend/common/imageOps.hpp"

#include "gpu/processors/tint.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/logging.hpp"

#include <iostream>

namespace VideoStitch {
namespace Core {

TintPreProcessor* TintPreProcessor::create(const Ptv::Value& config) {
  uint32_t color = 0xff00ff00;  // ABGR
  if (Parse::populateColor("ExprPreProcessor", config, "color", color, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  return new TintPreProcessor(color);
}

TintPreProcessor::TintPreProcessor(uint32_t color) : color(color) {}

TintPreProcessor::~TintPreProcessor() {}

Status TintPreProcessor::process(frameid_t /*frame*/, GPU::Surface& devBuffer, int64_t width, int64_t height,
                                 readerid_t /*inputId*/, GPU::Stream& stream) const {
  return tint(devBuffer, (unsigned)width, (unsigned)height, Image::RGBA::r(color), Image::RGBA::g(color),
              Image::RGBA::b(color), stream);
}

void TintPreProcessor::getDisplayName(std::ostream& os) const { os << "Procedural(P): Tint"; }
}  // namespace Core
}  // namespace VideoStitch
