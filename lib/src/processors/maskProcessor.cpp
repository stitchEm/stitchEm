// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "maskProcessor.hpp"

#include "gpu/processors/maskoverlay.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/logging.hpp"

#include <iostream>

namespace VideoStitch {
namespace Core {

MaskPreProcessor* MaskPreProcessor::create(const Ptv::Value& config) {
  uint32_t color = 0x300000ff;  // ABGR
  if (Parse::populateColor("ExprPreProcessor", config, "color", color, false) == Parse::PopulateResult_WrongType) {
    return NULL;
  }
  return new MaskPreProcessor(color);
}

MaskPreProcessor::MaskPreProcessor(uint32_t color) : color(color) {}

MaskPreProcessor::~MaskPreProcessor() {}

Status MaskPreProcessor::process(frameid_t /*frame*/, GPU::Surface& devBuffer, int64_t width, int64_t height,
                                 readerid_t /*inputId*/, GPU::Stream& stream) const {
  maskOverlay(devBuffer, (unsigned)width, (unsigned)height, color, stream);
  return Status::OK();
}

void MaskPreProcessor::getDisplayName(std::ostream& os) const { os << "Procedural(P): Mask"; }
}  // namespace Core
}  // namespace VideoStitch
